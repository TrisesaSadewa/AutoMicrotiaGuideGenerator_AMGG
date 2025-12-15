import cv2
import numpy as np
import os
import glob
import math
import torch
import sys
from pathlib import Path
from scipy.spatial import Delaunay
from ultralytics import YOLO
from stl import mesh
from segment_anything import sam_model_registry, SamPredictor

# --- CLINICAL CONFIGURATION (NAGATA STANDARDS) ---
CONF_THRESHOLD = 0.3
PIXELS_PER_MM = 10          # Scale: 10px = 1mm
BASE_THICKNESS = 2.0        # MINIMUM for rigidity (prevent warping)
SKELETON_THICKNESS = 4.0    # HIGH RELIEF for scalpel guiding (Total Height ~6mm)
STRUT_WIDTH_MM = 3.0        # MINIMUM wall width to hold sutures
SUTURE_SPACING_MM = 12.0

class HybridNagataGenerator:
    def __init__(self, yolo_path, sam_checkpoint=None):
        # 1. Load YOLO
        self.yolo_model = YOLO(yolo_path)
        print(f"✅ YOLO Model loaded: {yolo_path}")
        
        # 2. Load SAM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_predictor = self._load_sam(sam_checkpoint)
        
    def _load_sam(self, checkpoint_path):
        """Helper to load or download SAM"""
        if checkpoint_path is None:
            checkpoint_path = "sam_vit_b_01ec64.pth"
            
        if not os.path.exists(checkpoint_path):
            print(f"📥 Downloading SAM checkpoint to {checkpoint_path}...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            try:
                urllib.request.urlretrieve(url, checkpoint_path)
            except Exception as e:
                print(f"❌ Failed to download SAM: {e}")
                return None
                
        print(f"   Loading SAM from {checkpoint_path} on {self.device}...")
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return SamPredictor(sam)
        
    def detect_and_crop(self, image_path):
        results = self.yolo_model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)
        if not results or len(results[0].boxes) == 0: return None, None, None
        
        box = results[0].boxes.data[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box[:4])
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        pad = 60 
        px1, py1 = max(0, x1-pad), max(0, y1-pad)
        px2, py2 = min(w, x2+pad), min(h, y2+pad)
        
        crop = img[py1:py2, px1:px2]
        box_relative = np.array([x1-px1, y1-py1, x2-px1, y2-py1])
        
        return crop, (px1, py1), box_relative

    def extract_anatomy(self, image, prompt_box):
        """
        Generates ROBUST ANATOMICAL MASKS.
        Ensures parts are wide enough (3mm) to serve as physical cutting guides.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # --- 1. BASE PLATE (SAM) ---
        if self.sam_predictor:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None, point_labels=None,
                box=prompt_box[None, :], multimask_output=False,
            )
            mask_base = (masks[0] * 255).astype(np.uint8)
        else:
            blur_base = cv2.GaussianBlur(gray, (21, 21), 0)
            _, mask_base = cv2.threshold(blur_base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean Base
        cnts_base, _ = cv2.findContours(mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_base = np.zeros_like(gray)
        if cnts_base:
            base_cnt = max(cnts_base, key=cv2.contourArea)
            cv2.drawContours(mask_base, [base_cnt], -1, 255, -1)

        # --- 2. SKELETON (DoG + Thicken) ---
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        g1 = cv2.GaussianBlur(enhanced, (5, 5), 0)
        g2 = cv2.GaussianBlur(enhanced, (25, 25), 0)
        dog = cv2.subtract(g1, g2)
        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        
        # Lower threshold to find structure
        _, mask_raw = cv2.threshold(dog_norm, 35, 255, cv2.THRESH_BINARY)
        mask_raw = cv2.bitwise_and(mask_raw, mask_base)
        
        # CRITICAL: Widen the skeleton to meet Strut Width (3mm)
        # 3mm at 10px/mm = 30px width.
        # Morphological Closing bridges gaps, Dilate adds width.
        strut_px = int(STRUT_WIDTH_MM * PIXELS_PER_MM * 0.5) # Radius for kernel
        k_strut = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strut_px, strut_px))
        
        skeleton = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, k_strut, iterations=2)
        skeleton = cv2.dilate(skeleton, k_strut, iterations=1)
        
        # --- 3. CLASSIFY HELIX / ANTIHELIX ---
        mask_helix = np.zeros_like(gray)
        mask_antihelix = np.zeros_like(gray)
        
        # Rim Zone (Outer 15%)
        rim_width = int(w * 0.15) 
        k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_width, rim_width))
        eroded_base = cv2.erode(mask_base, k_rim, iterations=1)
        rim_zone = cv2.subtract(mask_base, eroded_base)
        
        cnts_ridges, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts_ridges:
            # Sort by length
            sorted_ridges = sorted(cnts_ridges, key=lambda x: cv2.arcLength(x, False), reverse=True)
            
            for cnt in sorted_ridges:
                length = cv2.arcLength(cnt, False)
                if length < 50: continue 
                
                # Check bounds
                x, y, cw, ch = cv2.boundingRect(cnt)
                if x<=5 or y<=5 or (x+cw)>=w-5 or (y+ch)>=h-5: continue 

                # Classify
                temp = np.zeros_like(gray)
                cv2.drawContours(temp, [cnt], -1, 255, -1)
                
                overlap = cv2.countNonZero(cv2.bitwise_and(temp, rim_zone))
                total = cv2.countNonZero(temp) + 1e-5
                
                if (overlap / total) > 0.3: 
                    cv2.drawContours(mask_helix, [cnt], -1, 255, -1)
                else:
                    cv2.drawContours(mask_antihelix, [cnt], -1, 255, -1)

        # Final Polish
        k_finish = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_helix = cv2.morphologyEx(mask_helix, cv2.MORPH_CLOSE, k_finish)
        mask_antihelix = cv2.morphologyEx(mask_antihelix, cv2.MORPH_CLOSE, k_finish)

        return mask_base, mask_helix, mask_antihelix

    def mask_to_stl(self, mask, output_path, thickness, base_height=0.0):
        """
        Converts mask to 3D STL.
        Ensures MANIFOLD geometry (Watertight) for slicing.
        """
        # Ensure mask is clean
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        all_points = []
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 20: continue
            
            # High res approximation for smooth curves
            epsilon = 0.001 * perimeter 
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for pt in approx: all_points.append(pt[0])
            
        if len(all_points) < 3: return None
        
        # 1. Round & Unique to fuse micro-gaps
        points_2d = np.array(all_points).astype(np.float32)
        points_2d = np.round(points_2d, 3) 
        points_2d = np.unique(points_2d, axis=0)
        
        if len(points_2d) < 3: return None
        
        try: tri = Delaunay(points_2d)
        except: return None
        
        valid_faces_2d = []
        for simplex in tri.simplices:
            center = np.mean(points_2d[simplex], axis=0).astype(int)
            # Check if triangle centroid is inside mask
            if 0 <= center[1] < mask.shape[0] and 0 <= center[0] < mask.shape[1]:
                if mask[center[1], center[0]] > 127: 
                    # Enforce CCW Winding
                    p1, p2, p3 = points_2d[simplex]
                    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    if cross < 0:
                        valid_faces_2d.append([simplex[0], simplex[2], simplex[1]])
                    else:
                        valid_faces_2d.append(simplex)
        
        if not valid_faces_2d: return None
        
        scale = 0.1 
        vertices_3d = []
        faces_3d = []
        n_pts = len(points_2d)
        
        # Vertices
        z_bot = base_height
        z_top = base_height + thickness
        
        for p in points_2d: vertices_3d.append([p[0]*scale, p[1]*scale, z_bot])
        for p in points_2d: vertices_3d.append([p[0]*scale, p[1]*scale, z_top])
        
        # Track edges
        existing_edges = set()
        
        for f in valid_faces_2d:
            p1, p2, p3 = f
            # Bottom (CW for Down normal)
            faces_3d.append([p1, p3, p2]) 
            # Top (CCW for Up normal)
            faces_3d.append([p1+n_pts, p2+n_pts, p3+n_pts])
            
            existing_edges.add((p1, p2))
            existing_edges.add((p2, p3))
            existing_edges.add((p3, p1))
            
        # Walls
        boundary_edges = []
        for (p1, p2) in existing_edges:
            if (p2, p1) not in existing_edges:
                boundary_edges.append((p1, p2))
                
        for p1, p2 in boundary_edges:
            b1, b2 = p1, p2
            t1, t2 = p1+n_pts, p2+n_pts
            # Quad faces pointing OUT
            faces_3d.append([t1, t2, b2])
            faces_3d.append([t1, b2, b1])

        obj = mesh.Mesh(np.zeros(len(faces_3d), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces_3d):
            for j in range(3): obj.vectors[i][j] = vertices_3d[f[j]]
        obj.save(output_path)
        return obj

    def create_hole_cutters(self, mask, output_path, z_start=-5, z_end=15):
        """Generates Suture Hole Cutters."""
        img = mask.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        # Robust Thinning
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0: break
        
        pts = cv2.findNonZero(skel)
        if pts is None: return
        pts = pts.squeeze()
        if len(pts.shape) == 1: pts = np.array([pts])
        
        spacing_px = int(SUTURE_SPACING_MM * PIXELS_PER_MM)
        final_holes = []
        if len(pts) > 0:
            sorted_pts = pts[pts[:, 1].argsort()]
            active_list = list(sorted_pts)
            while len(active_list) > 0:
                current = active_list.pop(0)
                final_holes.append(current)
                active_list = [p for p in active_list if np.linalg.norm(p - current) > spacing_px]

        all_faces, all_verts = [], []
        for cx, cy in final_holes:
            oval = []
            for t in np.linspace(0, 6.28, 20): 
                # 1.2 x 2.7 mm hole (scaled up slightly for tolerance)
                oval.append([cx + 6*math.cos(t), cy + 13.5*math.sin(t)]) 
            
            start_b = len(all_verts)
            for p in oval: all_verts.append([p[0], p[1], z_start])
            start_t = len(all_verts)
            for p in oval: all_verts.append([p[0], p[1], z_end])
            
            for i in range(20):
                ni = (i+1)%20
                b_i, b_ni = start_b+i, start_b+ni
                t_i, t_ni = start_t+i, start_t+ni
                all_faces.append([b_i, b_ni, t_i])
                all_faces.append([b_ni, t_ni, t_i])

        all_verts_np = np.array(all_verts) * 0.1 
        m = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(all_faces):
            for j in range(3): m.vectors[i][j] = all_verts_np[f[j]]
        m.save(output_path)

def run_batch():
    print("🚀 INITIALIZING NAGATA SURGICAL PIPELINE...")
    
    yolo_path = r"A:\PROJECT\AUTO MICROTIA\Ear-segmentation-ai\ear_segmentation\models\best_v4.pt"
    if not os.path.exists(yolo_path): return
    
    images = glob.glob("input/*.jpg") + glob.glob("input/*.png")
    output_dir = "surgical_guides_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    generator = HybridNagataGenerator(yolo_path)
    
    for img_path in images:
        name = Path(img_path).stem
        print(f"\n📸 Processing: {name}")
        
        crop, offset, box_relative = generator.detect_and_crop(img_path)
        if crop is None: continue
        
        # 1. Extract Anatomy
        mask_base, mask_helix, mask_antihelix = generator.extract_anatomy(crop, box_relative)
        
        # 2. Generate 3D Parts (Robust Thickness)
        # Base: 0 to 2mm
        m_base = generator.mask_to_stl(mask_base, f"{output_dir}/{name}_1_BASE.stl", 
                                     thickness=BASE_THICKNESS, base_height=0.0)
        
        # Skeleton: 2mm to 6mm (Sits on top)
        m_helix = generator.mask_to_stl(mask_helix, f"{output_dir}/{name}_2_HELIX.stl", 
                                      thickness=SKELETON_THICKNESS, base_height=BASE_THICKNESS)
        
        m_antihelix = generator.mask_to_stl(mask_antihelix, f"{output_dir}/{name}_3_ANTIHELIX.stl", 
                                          thickness=SKELETON_THICKNESS, base_height=BASE_THICKNESS)
        
        if m_base and m_helix and m_antihelix:
            # Combine All
            combined_data = np.concatenate([m_base.data, m_helix.data, m_antihelix.data])
            combined_mesh = mesh.Mesh(combined_data.copy())
            combined_mesh.save(f"{output_dir}/{name}_ALL_PARTS.stl")
            print(f"   ✅ Generated Combined Guide (Total Height: {BASE_THICKNESS + SKELETON_THICKNESS}mm)")
            
            # Holes (Cut through everything)
            skeleton_combined = cv2.bitwise_or(mask_helix, mask_antihelix)
            generator.create_hole_cutters(skeleton_combined, f"{output_dir}/{name}_holes.stl")

if __name__ == "__main__":
    run_batch()