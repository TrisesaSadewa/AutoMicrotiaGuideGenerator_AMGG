# Auto-Microtia Guide Generator (AMGG)

Automated Surgical Guide Generation for Microtia Reconstruction using Artificial Intelligence and Computer Vision.

📋 Overview

AMGG is a computer program designed to democratize high-precision surgical planning for Microtia reconstruction. By leveraging AI and Computer Vision, this tool converts standard 2D smartphone photos of a healthy ear into a biomechanically accurate 3D surgical guide (STL format) ready for 3D printing.

This project addresses the limitations of traditional manual 2D tracing methods (which lack depth) and the high cost/radiation risks of CT scans, providing a low-cost, accessible solution for surgeons in developing regions.

✨ Key Features

AI-Powered Segmentation: Utilizes Meta SAM (Segment Anything Model) and YOLOv8 to automatically isolate the healthy ear from complex backgrounds without manual tracing.

No CT Scan Required: Generates 3D topology from a single 2D image, reducing cost and radiation exposure for pediatric patients.

Nagata Technique Compliance: Automatically applies biomechanical standards:

Base Plate: 2.0 mm thickness (Simulating Ribs 6 & 7).

Helical Projection: 5.0 mm total height (Simulating Rib 8).

Contralateral Mirroring: Automatically flips the healthy ear geometry to create a guide for the affected side (supporting the 60-80% of cases that are Right-sided Microtia).

Automated Suture Windows: Generates 1.2mm x 2.7mm anchor holes for surgical needle guidance.

Watertight STL Output: Produces files ready for standard SLA/Resin 3D printers.

🛠️ Technology Stack

Language: Python 3.10

Segmentation: Meta SAM (Segment Anything Model), Ultralytics YOLOv8

Image Processing: OpenCV (cv2)

3D Mesh Generation: numpy-stl

Math/Logic: NumPy

⚙️ Installation

Clone the repository

git clone [https://github.com/TrisesaSadewa/AutoMicrotiaGuideGenerator_AMGG.git](https://github.com/TrisesaSadewa/AutoMicrotiaGuideGenerator_AMGG.git)
cd AutoMicrotiaGuideGenerator_AMGG


Create a virtual environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install dependencies

pip install -r requirements.txt
