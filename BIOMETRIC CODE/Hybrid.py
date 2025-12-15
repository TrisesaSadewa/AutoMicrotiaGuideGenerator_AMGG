import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Standard size for visualization (ORB is scale-invariant, so this is mainly for display consistency)
STANDARDIZED_SIZE = (128, 256) 
PROBE_FILES = ['2D EAR 1.jpg', '2D EAR 2.jpg', '2D EAR 3.jpg', '2D EAR 4.jpg', '2D EAR 5.jpg']
THRESHOLD_MATCHES = 15 # Minimum number of good matches required for a score of 1.0

# --- CORE PREPROCESSING FUNCTIONS ---

def get_skin_mask(image_color):
    """
    Creates a binary mask isolating skin areas using the YCrCb color space 
    to reduce hair and background noise before feature detection.
    """
    if image_color is None or image_color.size == 0:
        return np.zeros(image_color.shape[:2], dtype=np.uint8)

    # Convert to YCrCb 
    ycrcb = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)
    
    # Standard biometric range for skin tone
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    
    skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)
    
    return skin_mask

def preprocess_and_detect_features(image_gray):
    """
    Applies CLAHE (contrast normalization) and uses the ORB descriptor 
    to find keypoints (corners, edges) and their descriptors (features).
    """
    if image_gray is None or image_gray.size == 0:
        return None, None
    
    # 1. Contrast Enhancement (CLAHE for local contrast normalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_normalized_gray = clahe.apply(image_gray)
    
    # 2. Initialize ORB Detector (Feature Extractor)
    orb = cv2.ORB_create(nfeatures=2000) 
    
    # 3. Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(contrast_normalized_gray, None)
    
    return keypoints, descriptors

# --- TEMPLATE/GALLERY FUNCTIONS (Feature-based Enrollment) ---

def create_simulated_template_image(shape=(256, 128)):
    """
    SIMULATION: Creates a synthetic grayscale image that visually resembles 
    the EAR REF.jpg outline. This is ONLY used here to generate the GALLERY 
    FEATURES for the enrollment profile, as direct file reading is blocked.
    """
    h, w = shape[::-1] 
    template = np.full((h, w), 255, dtype=np.uint8) # White background
    
    # Draw curves similar to EAR REF.jpg (Left Ear Profile)
    # Outer Helix
    cv2.ellipse(template, (w // 2, h // 2), (w // 2 - 10, h // 2 - 20), 90, 0, 360, 0, 5)
    # Inner structures
    cv2.ellipse(template, (w // 2, h // 2), (w // 2 - 30, h // 2 - 50), 90, 0, 360, 0, 3)
    
    # Invert for white lines on black background 
    return cv2.bitwise_not(template)

def load_and_enroll_gallery():
    """
    Simulated Enrollment: Extracts features from the placeholder template 
    to create the final biometric profile (keypoints and descriptors).
    """
    print("Simulating Enrollment of Gallery Template features...")
    
    # Use the placeholder image (representing EAR REF.jpg)
    template_placeholder = create_simulated_template_image(shape=STANDARDIZED_SIZE)
    
    # Detect features from the template
    keypoints, descriptors = preprocess_and_detect_features(template_placeholder)

    if descriptors is None:
        raise ValueError("Could not extract features from the template placeholder.")

    # Return the template image (for display) and its unique feature set
    return template_placeholder, keypoints, descriptors

# --- BIOMETRIC SYSTEM CLASS (Feature-Based Matching) ---

class FeatureEarBiometrics:
    def __init__(self, gallery_template, gallery_kp, gallery_desc):
        self.gallery_template = gallery_template
        self.gallery_kp = gallery_kp
        self.gallery_desc = gallery_desc
        # BFMatcher (Brute-Force Matcher) for ORB descriptors (uses Hamming distance)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_probe(self, probe_path):
        """
        Processes a probe image, extracts features, and matches against the gallery.
        This function implicitly handles Segmentation (via Skin Masking) and Alignment 
        (via ORB's scale/rotation invariance).
        """
        # Load in Color for Masking
        img_color = cv2.imread(probe_path, cv2.IMREAD_COLOR)
        
        # --- Handle File Loading Failure (Fallback) ---
        if img_color is None:
             width, height = 400, 600
             # Synthetic noise fallback for image display
             img_color = np.random.randint(150, 200, size=(height, width, 3), dtype=np.uint8) 
             img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
             print(f"Warning: Using synthetic colored fallback for {probe_path}.")
        else:
             img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 1. Semantic Constraint (Skin Masking)
        skin_mask = get_skin_mask(img_color)
        masked_gray = cv2.bitwise_and(img_gray, img_gray, mask=skin_mask)
        
        # 2. Feature Extraction (Original and Mirrored)
        probe_kp_orig, probe_desc_orig = preprocess_and_detect_features(masked_gray)

        # Left/Right Mirroring: Flip the grayscale masked image horizontally
        masked_gray_mirrored = cv2.flip(masked_gray, 1)
        probe_kp_mirrored, probe_desc_mirrored = preprocess_and_detect_features(masked_gray_mirrored)

        # 3. Matching (Compare Original vs Mirrored Features)
        score_orig, matches_orig = self._match_and_score(probe_desc_orig)
        score_mirrored, matches_mirrored = self._match_and_score(probe_desc_mirrored)

        # Select the best match set
        if score_mirrored > score_orig:
            final_score = score_mirrored
            best_kp = probe_kp_mirrored
            is_mirrored = True
            best_matches = matches_mirrored
        else:
            final_score = score_orig
            best_kp = probe_kp_orig
            is_mirrored = False
            best_matches = matches_orig

        return final_score, img_gray, best_kp, is_mirrored, best_matches

    def _match_and_score(self, probe_desc):
        """Performs feature matching and calculates similarity score."""
        if probe_desc is None or self.gallery_desc is None or probe_desc.dtype != self.gallery_desc.dtype:
             return 0.0, []

        try:
            # Perform brute-force matching
            matches = self.matcher.match(self.gallery_desc, probe_desc)
            
            # Sort matches by distance (lower distance means better feature match)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Determine "good" matches using a distance threshold
            good_matches_count = len([m for m in matches if m.distance < 45]) 
            
            # Normalize score: Max similarity is 1.0 when good_matches_count >= THRESHOLD_MATCHES
            similarity_score = min(1.0, good_matches_count / THRESHOLD_MATCHES)
            
            # Return the similarity score and the best matches for visualization
            return similarity_score, matches[:good_matches_count] 

        except cv2.error:
            # Handle potential errors during matching if descriptor lists are oddly sized
            return 0.0, []

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    # --- STEP 1: LOAD AND ENROLL GALLERY TEMPLATE (FEATURE EXTRACTION) ---
    gallery_template_img, gallery_kp, gallery_desc = load_and_enroll_gallery() 

    # Initialize the Feature-Based Biometric System
    ear_system = FeatureEarBiometrics(gallery_template_img, gallery_kp, gallery_desc)
    
    print(f"--- LOCAL FEATURE BIOMETRIC SYSTEM ANALYSIS ({len(PROBE_FILES)} PROBES) ---")
    
    # Iterate through all probe images
    for i, probe_path in enumerate(PROBE_FILES):
        print(f"\n======================\nPROCESSING PROBE: {probe_path}")
        
        # --- STAGE 2 & 3: PROCESS PROBE (Detection, Alignment, and Recognition) ---
        final_score, probe_image_gray, best_kp, is_mirrored, best_matches = ear_system.process_probe(probe_path)
        
        print(f"1. Keypoints Found: {len(best_kp) if best_kp is not None else 0}")
        print(f"2. Image Orientation: {'Mirrored' if is_mirrored else 'Original'}")
        print(f"3. FINAL Similarity Score (ORB Matches): {final_score:.4f}")
        
        # Recognition decision logic
        if final_score >= 1.0:
            decision = f">> DECISION: MATCH (Score {final_score:.4f})"
        else:
            decision = f">> DECISION: NO MATCH (Score {final_score:.4f})"
        print(decision)
        
        # --- VISUALIZATION ---
        
        # Draw the matches on the image
        img_matches = cv2.drawMatches(
            ear_system.gallery_template, ear_system.gallery_kp, 
            cv2.flip(probe_image_gray, 1) if is_mirrored else probe_image_gray, best_kp, 
            best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        fig.suptitle(f"PROBE {i+1}: {probe_path} | {decision} | Best Match: {'MIRRORED' if is_mirrored else 'ORIGINAL'}", 
                     fontsize=14, fontweight='bold')
        
        # Convert image from BGR to RGB for correct matplotlib display
        axes.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        axes.set_title(f"Feature Matching: Gallery (Left) vs Probe (Right)")
        axes.axis('off')
        
    plt.show()
