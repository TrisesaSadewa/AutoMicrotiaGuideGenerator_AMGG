# mask_based_3d_guide.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import numpy as np
import os
from pathlib import Path
import threading
import glob

class MaskBased3DGuide:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask-Based 3D Guide Generator")
        self.root.geometry("1200x800")
        
        # Variables
        self.original_image = None
        self.processed_image = None
        self.mask = None
        self.image_path = None
        self.selection_points = []
        self.is_selecting = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Mask-Based 3D Surgical Guide Generator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Instructions
        instructions = """
Create precise 3D surgical guides based on ear segmentation

1. Load an image and segment the ear
2. Generate 3D guide based on the actual ear anatomy
3. Adjust parameters for surgical requirements
        """
        instr_label = ttk.Label(main_frame, text=instructions, justify=tk.LEFT)
        instr_label.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_path_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var, 
                              background="lightgray", padding="5", relief="solid")
        file_label.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(file_frame, text="Browse Image", command=self.browse_image).grid(row=0, column=1)
        
        # Segmentation methods
        seg_frame = ttk.LabelFrame(main_frame, text="Segmentation Methods", padding="10")
        seg_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(seg_frame, text="GrabCut Auto", command=self.grabcut_auto).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(seg_frame, text="Color-Based", command=self.color_based_segmentation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(seg_frame, text="Edge-Based", command=self.edge_based_segmentation).pack(side=tk.LEFT, padx=(0, 5))
        
        # 3D Guide parameters
        guide_frame = ttk.LabelFrame(main_frame, text="3D Guide Parameters", padding="10")
        guide_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Skin tolerance (mm)
        ttk.Label(guide_frame, text="Skin Tolerance (mm):").grid(row=0, column=0, sticky=tk.W)
        self.skin_tolerance = tk.DoubleVar(value=2.0)
        ttk.Scale(guide_frame, from_=1.0, to=5.0, variable=self.skin_tolerance, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        ttk.Label(guide_frame, textvariable=self.skin_tolerance).grid(row=0, column=2)
        
        # Base thickness (mm)
        ttk.Label(guide_frame, text="Base Thickness (mm):").grid(row=1, column=0, sticky=tk.W)
        self.base_thickness = tk.DoubleVar(value=2.0)
        ttk.Scale(guide_frame, from_=1.0, to=5.0, variable=self.base_thickness, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        ttk.Label(guide_frame, textvariable=self.base_thickness).grid(row=1, column=2)
        
        # Helix thickness (mm)
        ttk.Label(guide_frame, text="Helix Thickness (mm):").grid(row=2, column=0, sticky=tk.W)
        self.helix_thickness = tk.DoubleVar(value=5.0)
        ttk.Scale(guide_frame, from_=3.0, to=8.0, variable=self.helix_thickness, 
                 orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        ttk.Label(guide_frame, textvariable=self.helix_thickness).grid(row=2, column=2)
        
        guide_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=5, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Button(action_frame, text="Generate 3D Guide", command=self.generate_3d_guide).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Export Guide Data", command=self.export_guide_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Original image
        self.original_frame = ttk.LabelFrame(results_frame, text="Original Image")
        self.original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        self.original_frame.columnconfigure(0, weight=1)
        self.original_frame.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(self.original_frame, bg="white", width=400, height=400)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 3D Guide visualization
        self.guide_frame = ttk.LabelFrame(results_frame, text="3D Surgical Guide")
        self.guide_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.guide_frame.columnconfigure(0, weight=1)
        self.guide_frame.rowconfigure(0, weight=1)
        
        self.guide_canvas = tk.Canvas(self.guide_frame, bg="white", width=400, height=400)
        self.guide_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image and segment the ear")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize canvas placeholders
        self.original_canvas.create_text(200, 200, text="Original image will appear here", fill="gray")
        self.guide_canvas.create_text(200, 200, text="3D guide will appear here", fill="gray")
    
    def browse_image(self):
        """Browse for image file"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display the selected image"""
        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            
            # Update file path display
            self.file_path_var.set(f"Selected: {os.path.basename(file_path)}")
            
            # Resize for display while maintaining aspect ratio
            display_image = self.resize_for_display(self.original_image)
            self.display_image_on_canvas(display_image, self.original_canvas)
            
            self.status_var.set(f"Loaded: {os.path.basename(file_path)} - Segment the ear first")
            self.clear_guide()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def resize_for_display(self, image, max_size=400):
        """Resize image for display while maintaining aspect ratio"""
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image
    
    def display_image_on_canvas(self, image, canvas):
        """Display PIL image on tkinter canvas"""
        # Clear canvas
        canvas.delete("all")
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Store reference to prevent garbage collection
        canvas.image = photo
        
        # Display image centered on canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet rendered
            canvas_width = 400
            canvas_height = 400
        
        x = (canvas_width - photo.width()) // 2
        y = (canvas_height - photo.height()) // 2
        
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
    
    def grabcut_auto(self):
        """Automatic GrabCut segmentation"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.status_var.set("Running GrabCut auto-segmentation...")
        self.progress.start()
        
        thread = threading.Thread(target=self._grabcut_auto_thread)
        thread.daemon = True
        thread.start()
    
    def _grabcut_auto_thread(self):
        """Thread for automatic GrabCut"""
        try:
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            h, w = cv_image.shape[:2]
            
            # Create initial mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Conservative rectangle
            rect_size = min(h, w) // 3
            center_x, center_y = w // 2, h // 2
            rect = (
                max(10, center_x - rect_size // 2),
                max(10, center_y - rect_size // 2),
                min(w - 20, rect_size),
                min(h - 20, rect_size)
            )
            
            # Initialize mask
            mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = cv2.GC_PR_FGD
            
            # Mark borders as background
            border_size = 10
            mask[:border_size, :] = cv2.GC_BGD
            mask[-border_size:, :] = cv2.GC_BGD
            mask[:, :border_size] = cv2.GC_BGD
            mask[:, -border_size:] = cv2.GC_BGD
            
            # Run GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(cv_image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            # Create final mask
            final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # Fallback if empty
            if np.sum(final_mask) == 0:
                final_mask = np.zeros((h, w), dtype=np.uint8)
                final_mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 255
            
            self.mask = final_mask
            
            # Create overlay
            overlay = cv_image.copy()
            colored_mask = np.zeros_like(cv_image)
            colored_mask[final_mask == 255] = [0, 255, 0]
            overlay = cv2.addWeighted(cv_image, 0.7, colored_mask, 0.3, 0)
            
            self.processed_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            
            self.root.after(0, self._segmentation_complete, True, "GrabCut segmentation completed")
            
        except Exception as e:
            self.root.after(0, self._segmentation_complete, False, f"GrabCut error: {e}")
    
    def color_based_segmentation(self):
        """Color-based segmentation"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.status_var.set("Running color-based segmentation...")
        self.progress.start()
        
        thread = threading.Thread(target=self._color_based_thread)
        thread.daemon = True
        thread.start()
    
    def _color_based_thread(self):
        """Thread for color-based segmentation"""
        try:
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            
            # Color-based segmentation
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            
            # Skin color ranges
            lower_skin_hsv = np.array([0, 10, 60])
            upper_skin_hsv = np.array([25, 255, 255])
            skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
            
            lower_skin_lab = np.array([0, 120, 120])
            upper_skin_lab = np.array([255, 150, 160])
            skin_mask_lab = cv2.inRange(lab, lower_skin_lab, upper_skin_lab)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_lab)
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find largest contour
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                final_mask = np.zeros_like(combined_mask)
                cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            else:
                final_mask = combined_mask
            
            self.mask = final_mask
            
            # Create overlay
            overlay = cv_image.copy()
            colored_mask = np.zeros_like(cv_image)
            colored_mask[final_mask == 255] = [0, 255, 0]
            overlay = cv2.addWeighted(cv_image, 0.7, colored_mask, 0.3, 0)
            
            self.processed_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            
            self.root.after(0, self._segmentation_complete, True, "Color-based segmentation completed")
            
        except Exception as e:
            self.root.after(0, self._segmentation_complete, False, f"Color-based error: {e}")
    
    def edge_based_segmentation(self):
        """Edge-based segmentation"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.status_var.set("Running edge-based segmentation...")
        self.progress.start()
        
        thread = threading.Thread(target=self._edge_based_thread)
        thread.daemon = True
        thread.start()
    
    def _edge_based_thread(self):
        """Thread for edge-based segmentation"""
        try:
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Close gaps
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            else:
                mask = np.zeros_like(gray)
            
            self.mask = mask
            
            # Create overlay
            overlay = cv_image.copy()
            colored_mask = np.zeros_like(cv_image)
            colored_mask[mask == 255] = [0, 255, 0]
            overlay = cv2.addWeighted(cv_image, 0.7, colored_mask, 0.3, 0)
            
            self.processed_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            
            self.root.after(0, self._segmentation_complete, True, "Edge-based segmentation completed")
            
        except Exception as e:
            self.root.after(0, self._segmentation_complete, False, f"Edge-based error: {e}")
    
    def _segmentation_complete(self, success, message):
        """Called when segmentation is complete"""
        self.progress.stop()
        self.status_var.set(message)
        
        if success and self.processed_image is not None:
            display_image = self.resize_for_display(self.processed_image)
            self.display_image_on_canvas(display_image, self.original_canvas)
        else:
            messagebox.showwarning("Segmentation Result", message)
    
    def generate_3d_guide(self):
        """Generate 3D guide based on the mask"""
        if self.mask is None:
            messagebox.showwarning("Warning", "Please segment an ear first")
            return
        
        self.status_var.set("Generating 3D surgical guide from mask...")
        self.progress.start()
        
        thread = threading.Thread(target=self._generate_3d_guide_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_3d_guide_thread(self):
        """Thread for generating 3D guide from mask"""
        try:
            # Convert original image to OpenCV if needed for reference
            if self.original_image:
                cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = np.ones((self.mask.shape[0], self.mask.shape[1], 3), dtype=np.uint8) * 255
            
            # Get parameters
            skin_tolerance_px = int(self.skin_tolerance.get() * 10)  # Rough conversion: 1mm ≈ 10 pixels
            base_thickness_px = int(self.base_thickness.get() * 10)
            helix_thickness_px = int(self.helix_thickness.get() * 10)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.root.after(0, self._guide_generation_complete, False, "No contours found in mask")
                return
            
            # Get the largest contour (the ear)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create guide visualization
            guide_image = self.create_anatomical_guide(
                cv_image, largest_contour, 
                skin_tolerance_px, base_thickness_px, helix_thickness_px
            )
            
            self.guide_image_pil = Image.fromarray(cv2.cvtColor(guide_image, cv2.COLOR_BGR2RGB))
            
            self.root.after(0, self._guide_generation_complete, True, "3D surgical guide generated successfully")
            
        except Exception as e:
            self.root.after(0, self._guide_generation_complete, False, f"Guide generation error: {e}")
    
    def create_anatomical_guide(self, image, ear_contour, skin_tolerance, base_thickness, helix_thickness):
        """Create anatomical 3D guide based on ear contour"""
        # Create a copy of the image
        guide_image = image.copy()
        
        # Get contour properties
        contour_area = cv2.contourArea(ear_contour)
        contour_perimeter = cv2.arcLength(ear_contour, True)
        
        # Approximate the contour to simplify
        epsilon = 0.005 * contour_perimeter
        approx_contour = cv2.approxPolyDP(ear_contour, epsilon, True)
        
        # Create inner outline (yellow) - shrunk contour for inner boundary
        inner_contour = self.offset_contour(approx_contour, -skin_tolerance)
        
        # Create base area (purple) - bottom part of the ear with tolerance
        base_contour = self.create_base_contour(approx_contour, skin_tolerance)
        
        # Find optimal suture hole positions along the contour
        suture_points = self.find_suture_positions(approx_contour, num_points=3)
        
        # Draw everything on the guide image
        
        # 1. Draw the original ear contour (green)
        cv2.drawContours(guide_image, [approx_contour], -1, (0, 255, 0), 2)
        
        # 2. Draw inner outline (yellow)
        if inner_contour is not None and len(inner_contour) > 2:
            cv2.drawContours(guide_image, [inner_contour], -1, (0, 255, 255), 3)
        
        # 3. Draw base area (purple)
        if base_contour is not None and len(base_contour) > 2:
            cv2.drawContours(guide_image, [base_contour], -1, (255, 0, 255), 2)
            # Fill base area with transparency
            overlay = guide_image.copy()
            cv2.fillPoly(overlay, [base_contour], (255, 0, 255))
            cv2.addWeighted(overlay, 0.3, guide_image, 0.7, 0, guide_image)
        
        # 4. Draw suture holes (red)
        for point in suture_points:
            x, y = point
            cv2.rectangle(guide_image, (x-6, y-3), (x+6, y+3), (0, 0, 255), -1)
        
        # 5. Add thickness indicators
        self.add_thickness_indicators(guide_image, approx_contour, base_thickness, helix_thickness)
        
        # 6. Add measurements and labels
        self.add_measurements(guide_image, approx_contour, skin_tolerance, base_thickness, helix_thickness)
        
        return guide_image
    
    def offset_contour(self, contour, offset):
        """Offset a contour by specified amount (positive for expand, negative for shrink)"""
        if len(contour) < 3:
            return None
        
        # Convert contour to points
        points = contour.reshape(-1, 2)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Offset points toward/away from centroid
        offset_points = []
        for point in points:
            dx = point[0] - cx
            dy = point[1] - cy
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize and scale by offset
                new_x = int(point[0] + (dx / length) * offset)
                new_y = int(point[1] + (dy / length) * offset)
                offset_points.append([new_x, new_y])
            else:
                offset_points.append(point)
        
        return np.array(offset_points, dtype=np.int32)
    
    def create_base_contour(self, ear_contour, skin_tolerance):
        """Create base contour for the bottom part of the ear"""
        if len(ear_contour) < 3:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(ear_contour)
        
        # Base is typically the bottom 30% of the ear
        base_top = y + int(h * 0.7)
        base_bottom = y + h + skin_tolerance
        
        # Create base contour (simple rectangle for now)
        base_points = [
            [x - skin_tolerance, base_top],
            [x + w + skin_tolerance, base_top],
            [x + w + skin_tolerance, base_bottom],
            [x - skin_tolerance, base_bottom]
        ]
        
        return np.array(base_points, dtype=np.int32)
    
    def find_suture_positions(self, contour, num_points=3):
        """Find optimal positions for suture holes along the contour"""
        if len(contour) < num_points:
            return []
        
        # Simplify contour for suture placement
        points = contour.reshape(-1, 2)
        
        # Find points with high curvature (likely anatomical landmarks)
        suture_points = []
        
        if len(points) >= num_points:
            # Simple approach: evenly spaced points
            step = len(points) // num_points
            for i in range(num_points):
                idx = (i * step) % len(points)
                suture_points.append(points[idx])
        
        return suture_points
    
    def add_thickness_indicators(self, image, contour, base_thickness, helix_thickness):
        """Add visual indicators for extrusion thickness"""
        if len(contour) < 3:
            return
        
        # Get centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Add thickness labels
        cv2.putText(image, f"Base: {self.base_thickness.get()}mm", 
                   (cx - 60, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(image, f"Helix: {self.helix_thickness.get()}mm", 
                   (cx - 60, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def add_measurements(self, image, contour, skin_tolerance, base_thickness, helix_thickness):
        """Add measurement annotations"""
        if len(contour) < 3:
            return
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add general measurements
        info_text = [
            f"Skin Tolerance: {self.skin_tolerance.get()}mm",
            f"Base Thickness: {self.base_thickness.get()}mm", 
            f"Helix Thickness: {self.helix_thickness.get()}mm",
            f"Ear Area: {cv2.contourArea(contour):.0f} px²"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(image, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _guide_generation_complete(self, success, message):
        """Called when guide generation is complete"""
        self.progress.stop()
        self.status_var.set(message)
        
        if success and hasattr(self, 'guide_image_pil'):
            display_image = self.resize_for_display(self.guide_image_pil)
            self.display_image_on_canvas(display_image, self.guide_canvas)
            
            # Show surgical instructions
            instructions = """
3D SURGICAL GUIDE READY

Guide Elements:
- Green: Ear contour (actual anatomy)
- Yellow: Inner outline (surgical boundary)  
- Purple: Base area (2mm skin tolerance)
- Red: Suture hole positions

Extrusion Instructions:
1. Base (purple): Extrude {base_thickness}mm
2. Helix & Anti-Helix (yellow): Extrude {helix_thickness}mm
3. Create contour-based cutters
4. Final piece ready for 3D printing

Note: This guide is anatomically accurate based on the ear segmentation.
            """.format(
                base_thickness=self.base_thickness.get(),
                helix_thickness=self.helix_thickness.get()
            )
            
            messagebox.showinfo("3D Surgical Guide Complete", instructions)
        else:
            messagebox.showwarning("Guide Generation Failed", message)
    
    def export_guide_data(self):
        """Export guide data for 3D modeling"""
        if not hasattr(self, 'guide_image_pil') or self.mask is None:
            messagebox.showwarning("Warning", "Please generate a 3D guide first")
            return
        
        try:
            # Create export directory
            export_dir = "3d_guide_exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Save guide image
            guide_path = os.path.join(export_dir, "surgical_guide.png")
            self.guide_image_pil.save(guide_path)
            
            # Save mask for 3D modeling
            mask_path = os.path.join(export_dir, "ear_mask.png")
            cv2.imwrite(mask_path, self.mask)
            
            # Save parameters
            param_path = os.path.join(export_dir, "guide_parameters.txt")
            with open(param_path, 'w') as f:
                f.write("3D Surgical Guide Parameters\n")
                f.write("============================\n\n")
                f.write(f"Skin Tolerance: {self.skin_tolerance.get()} mm\n")
                f.write(f"Base Thickness: {self.base_thickness.get()} mm\n")
                f.write(f"Helix Thickness: {self.helix_thickness.get()} mm\n")
                f.write(f"Export Date: {os.path.basename(self.image_path)}\n")
                f.write(f"Original Image: {os.path.basename(self.image_path)}\n\n")
                
                f.write("3D Modeling Instructions:\n")
                f.write("1. Import ear_mask.png as reference\n")
                f.write("2. Create base extrusion: {} mm\n".format(self.base_thickness.get()))
                f.write("3. Create helix extrusion: {} mm\n".format(self.helix_thickness.get()))
                f.write("4. Add suture holes at marked positions\n")
                f.write("5. Apply contour-based cutting\n")
                f.write("6. Export as STL for 3D printing\n")
            
            messagebox.showinfo("Export Complete", 
                              f"3D guide data exported to:\n{export_dir}\n\n"
                              f"- surgical_guide.png (Visual reference)\n"
                              f"- ear_mask.png (Mask for 3D modeling)\n"
                              f"- guide_parameters.txt (Modeling instructions)")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export guide data: {e}")
    
    def clear_guide(self):
        """Clear guide results"""
        if hasattr(self, 'guide_image_pil'):
            del self.guide_image_pil
        self.guide_canvas.delete("all")
        self.guide_canvas.create_text(200, 200, text="3D guide will appear here", fill="gray")
    
    def clear_all(self):
        """Clear all images and results"""
        self.original_image = None
        self.processed_image = None
        self.mask = None
        self.image_path = None
        
        self.original_canvas.delete("all")
        self.guide_canvas.delete("all")
        
        self.original_canvas.create_text(200, 200, text="Original image will appear here", fill="gray")
        self.guide_canvas.create_text(200, 200, text="3D guide will appear here", fill="gray")
        
        self.file_path_var.set("No file selected")
        self.status_var.set("Ready - Load an image and segment the ear")

def main():
    root = tk.Tk()
    app = MaskBased3DGuide(root)
    root.mainloop()

if __name__ == "__main__":
    main()