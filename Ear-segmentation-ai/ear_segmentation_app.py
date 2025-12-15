# microtia_guide_app.py
from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
from pathlib import Path
import trimesh
from stl import mesh
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import Delaunay

class MicrotiaGuideGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Microtia guide parameters (in mm)
        self.base_thickness = 2.0
        self.helix_thickness = 5.0
        self.skin_tolerance = 2.0
        self.suture_hole_size = (1.2, 2.7)  # width, height in mm
        
    def load_model(self):
        """Load the YOLO segmentation model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return False

    def preprocess_mask(self, mask):
        """Preprocess mask for microtia guide generation"""
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        return cleaned_mask

    def create_inner_contour(self, mask, tolerance_pixels):
        """Create inner contour with yellow outline and skin tolerance"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
            
        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create inner contour with tolerance
        inner_contour = self.offset_contour(main_contour, -tolerance_pixels)
        
        return main_contour, inner_contour

    def offset_contour(self, contour, offset):
        """Offset contour by specified distance"""
        # Convert contour to mask
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Apply offset
        kernel_size = abs(offset) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if offset > 0:
            processed_mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            processed_mask = cv2.erode(mask, kernel, iterations=1)
            
        # Extract new contour
        new_contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(new_contours, key=cv2.contourArea) if new_contours else contour

    def create_suture_holes(self, contour, image_shape):
        """Create suture holes at curvature points"""
        holes = []
        
        # Simplify contour to find curvature points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx_contour) >= 3:
            # Find points with high curvature (simplified approach)
            for i in range(len(approx_contour)):
                pt1 = approx_contour[i-1][0]
                pt2 = approx_contour[i][0]
                pt3 = approx_contour[(i+1)%len(approx_contour)][0]
                
                # Calculate angle (simplified curvature estimation)
                vec1 = pt1 - pt2
                vec2 = pt3 - pt2
                angle = np.arctan2(vec1[1], vec1[0]) - np.arctan2(vec2[1], vec2[0])
                
                if abs(angle) > 0.5:  # High curvature point
                    hole_center = pt2
                    holes.append(hole_center)
                    
                    # Limit to 2 main holes
                    if len(holes) >= 2:
                        break
        
        return holes

    def create_3d_guide(self, mask, output_path, scale_factor=0.1):
        """Create 3D microtia guide from mask"""
        try:
            # Preprocess mask
            processed_mask = self.preprocess_mask(mask)
            
            # Calculate scale (pixels to mm)
            pixels_per_mm = 10  # Adjust based on your image resolution
            tolerance_pixels = int(self.skin_tolerance * pixels_per_mm)
            
            # Create contours
            outer_contour, inner_contour = self.create_inner_contour(processed_mask, tolerance_pixels)
            if outer_contour is None:
                print("❌ No contour found in mask")
                return False
            
            # Create suture holes
            suture_holes = self.create_suture_holes(inner_contour, processed_mask.shape)
            
            # Generate 3D mesh
            mesh_data = self.generate_3d_mesh(inner_contour, outer_contour, suture_holes, scale_factor)
            
            # Save STL file
            mesh_data.save(output_path)
            
            # Generate visualization
            self.create_guide_visualization(processed_mask, outer_contour, inner_contour, suture_holes, output_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating 3D guide: {e}")
            return False

    def generate_3d_mesh(self, inner_contour, outer_contour, suture_holes, scale_factor):
        """Generate 3D mesh from contours"""
        # Convert contours to points
        inner_points = inner_contour.reshape(-1, 2).astype(float) * scale_factor
        outer_points = outer_contour.reshape(-1, 2).astype(float) * scale_factor
        
        # Create base layer (purple, 2mm)
        base_vertices = []
        base_faces = []
        
        # Add z-coordinate
        inner_points_3d = np.hstack([inner_points, np.zeros((len(inner_points), 1))])
        inner_points_3d_top = np.hstack([inner_points, np.full((len(inner_points), 1), self.base_thickness)])
        
        # Create base mesh (simplified)
        vertices = np.vstack([inner_points_3d, inner_points_3d_top])
        
        # Create simple triangular mesh
        tri = Delaunay(inner_points)
        faces = tri.simplices
        
        # Combine with top faces
        faces = np.vstack([faces, faces + len(inner_points)])
        
        # Create mesh
        guide_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                guide_mesh.vectors[i][j] = vertices[face[j]]
        
        return guide_mesh

    def create_guide_visualization(self, original_mask, outer_contour, inner_contour, suture_holes, output_path):
        """Create 2D visualization of the microtia guide"""
        # Create visualization image
        vis_image = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw contours
        cv2.drawContours(vis_image, [outer_contour], -1, (255, 0, 0), 2)  # Blue - outer contour
        cv2.drawContours(vis_image, [inner_contour], -1, (0, 255, 255), 2)  # Yellow - inner contour
        
        # Draw suture holes
        for hole in suture_holes:
            cv2.rectangle(vis_image, 
                         (int(hole[0] - 5), int(hole[1] - 5)),
                         (int(hole[0] + 5), int(hole[1] + 5)),
                         (0, 0, 255), -1)  # Red suture holes
        
        # Add labels
        cv2.putText(vis_image, "Outer Contour (Base)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_image, "Inner Contour (Helix/Anti-Helix)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, "Suture Holes", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save visualization
        vis_path = output_path.replace('.stl', '_visualization.jpg')
        cv2.imwrite(vis_path, vis_image)
        print(f"📊 Guide visualization saved: {vis_path}")

class YOLOEarSegmenter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.guide_generator = MicrotiaGuideGenerator(model_path)
        self.load_model()
    
    def load_model(self):
        """Load the YOLO segmentation model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return False
    
    def process_image(self, image_path, output_dir="microtia_output_2", conf=0.5, generate_3d=True):
        """Process a single image and generate microtia guide"""
        if self.model is None:
            print("❌ Model not loaded")
            return None
        
        try:
            # Run inference
            results = self.model.predict(image_path, imgsz=640, conf=conf, verbose=False)
            
            if len(results) == 0:
                return None
            
            result = results[0]
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            
            # Check if any ears were detected
            ear_detected = len(result.boxes) > 0 if result.boxes is not None else False
            
            # Save annotated image
            annotated_img = result.plot()
            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_img)
            
            # Process mask and generate microtia guide
            mask_path = None
            guide_path = None
            
            if ear_detected and result.masks is not None and len(result.masks) > 0:
                # Get the first mask
                mask = result.masks[0].data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, mask)
                
                # Generate 3D microtia guide
                if generate_3d:
                    guide_path = os.path.join(output_dir, f"{base_name}_microtia_guide.stl")
                    guide_success = self.guide_generator.create_3d_guide(mask, guide_path)
                    
                    if guide_success:
                        print(f"🦻 3D Microtia guide generated: {guide_path}")
                    else:
                        guide_path = None
                        print(f"❌ Failed to generate 3D guide for {base_name}")
            
            # Calculate area percentage
            area_percent = 0
            if ear_detected and result.masks is not None:
                total_pixels = mask.shape[0] * mask.shape[1]
                ear_pixels = np.sum(mask > 0)
                area_percent = (ear_pixels / total_pixels) * 100
            
            return {
                'success': True,
                'ear_detected': ear_detected,
                'area_percent': area_percent,
                'annotated_path': annotated_path,
                'mask_path': mask_path,
                'guide_path': guide_path,
                'num_detections': len(result.boxes) if result.boxes is not None else 0
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def batch_process(self, input_dir, output_dir="microtia_batch_output", conf=0.5, max_images=None, generate_3d=True):
        """Process all images in a directory and generate microtia guides"""
        if self.model is None:
            print("❌ Model not loaded")
            return
        
        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if not image_files:
            print(f"❌ No images found in {input_dir}")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"🔍 Processing {len(image_files)} images from {input_dir}")
        print(f"🦻 3D Guide Generation: {'ENABLED' if generate_3d else 'DISABLED'}")
        
        results = []
        for i, img_path in enumerate(image_files):
            print(f"\n📸 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            result = self.process_image(img_path, output_dir, conf, generate_3d)
            if result:
                status = "✓" if result['ear_detected'] else "✗"
                area_info = f"{result['area_percent']:.1f}%" if result['ear_detected'] else "-"
                detections = result['num_detections'] if result['ear_detected'] else 0
                
                print(f"   {status} Ear detected: {result['ear_detected']}")
                print(f"   📐 Area coverage: {area_info}")
                print(f"   🔍 Objects found: {detections}")
                
                if result['ear_detected'] and generate_3d:
                    if result['guide_path']:
                        print(f"   🦻 3D Guide: ✓ Generated")
                    else:
                        print(f"   🦻 3D Guide: ✗ Failed")
                
                results.append((img_path, result['ear_detected'], result['area_percent'], 
                              result['num_detections'], result.get('guide_path', None)))
            else:
                print(f"   ✗ Processing failed")
                results.append((img_path, False, 0, 0, None))
        
        # Calculate statistics
        detected_count = sum(1 for _, detected, _, _, _ in results if detected)
        detection_rate = (detected_count / len(results)) * 100
        total_detections = sum(detections for _, _, _, detections, _ in results)
        guides_generated = sum(1 for _, _, _, _, guide in results if guide is not None)
        
        print(f"\n📊 MICROTIA GUIDE GENERATION RESULTS:")
        print(f"   Total images processed: {len(results)}")
        print(f"   Ears detected: {detected_count}")
        print(f"   Detection rate: {detection_rate:.1f}%")
        print(f"   Total ear objects found: {total_detections}")
        if generate_3d:
            print(f"   3D guides generated: {guides_generated}")
        print(f"   Results saved to: {output_dir}")
        
        # Save results summary
        self.save_summary(results, output_dir, generate_3d)
        return results
    
    def save_summary(self, results, output_dir, generate_3d):
        """Save processing summary to CSV"""
        summary_path = os.path.join(output_dir, "microtia_processing_summary.csv")
        with open(summary_path, 'w') as f:
            if generate_3d:
                f.write("Image,Ear_Detected,Area_Percent,Num_Detections,Guide_Generated\n")
                for img_path, detected, area, detections, guide in results:
                    guide_status = "Yes" if guide else "No"
                    f.write(f"{os.path.basename(img_path)},{detected},{area:.2f},{detections},{guide_status}\n")
            else:
                f.write("Image,Ear_Detected,Area_Percent,Num_Detections\n")
                for img_path, detected, area, detections, _ in results:
                    f.write(f"{os.path.basename(img_path)},{detected},{area:.2f},{detections}\n")
        print(f"📄 Summary saved: {summary_path}")

def print_instructions():
    """Print 3D guide generation instructions"""
    print("\n" + "="*60)
    print("🦻 3D MICROTIA GUIDE GENERATION INSTRUCTIONS")
    print("="*60)
    print("1. ✅ Pastikan posisi telinga benar-benar tegak lurus")
    print("2. 🟡 Buat outline dalam yang berwarna kuning")
    print("3. 📏 Beri toleransi kulit 2mm untuk area base")
    print("4. 🕳️  Buat lubang jahit: 1.2mm x 2.7mm di titik lengkung")
    print("5. 📐 Extrude bagian:")
    print("   - Base (ungu) = 2mm")
    print("   - Helix & Anti Helix (kuning) = 5mm")
    print("6. ✂️  Buat pemotong sesuai contour telinga")
    print("7. 🖨️  Final piece siap untuk print")
    print("="*60 + "\n")

def main():
    print("🚀 MICROTIA SURGICAL GUIDE GENERATION APP")
    print_instructions()
    
    # Initialize with your YOLO model
    model_path = r"A:\PROJECT\AUTO MICROTIA\Ear-segmentation-ai\ear_segmentation\models\best_v4.pt"
    segmenter = YOLOEarSegmenter(model_path)
    
    if segmenter.model is None:
        print("❌ Failed to load YOLO model")
        return
    
    # Test the model on your dataset
    input_directory = "input_2"  
    if os.path.exists(input_directory):
        print(f"🎯 Processing directory: {input_directory}")
        
        # Process with 3D guide generation
        results = segmenter.batch_process(
            input_directory, 
            "microtia_output_2", 
            conf=0.5,
            generate_3d=True,
            max_images=None  # Process all images
        )
        
        print("\n✅ Processing completed!")
        print("💡 Check the output directory for:")
        print("   - Annotated detection images")
        print("   - Ear segmentation masks") 
        print("   - 3D microtia guide STL files")
        print("   - Guide visualizations")
        print("   - Processing summary CSV")
        
    else:
        print(f"❌ Directory not found: {input_directory}")
        print("💡 Please update the input_directory path")

if __name__ == "__main__":
    main()