# HSST Algorithm (C++ Implementation)

This folder contains the **C++ implementation** of the proposed **HSST (Hybrid Saliency and Stability Thresholding)** algorithm and traditional visual comparison methods for fabric stain detection.

The implementation includes:

- **Gabor Parameter Optimization:** Using the IVY algorithm to find optimal Gabor filter parameters for diverse fabric textures.  
- **Hybrid Saliency Fusion:** Inspired by human visual contrast perception, integrating global and local contrast cues to highlight stains.  
- **Stability-Driven Thresholding:** Adaptive thresholding based on segmentation stability for robust stain detection.  
- **Comparison Methods:** The implementaion of the comparison traditional visual methods including GFC, BAS-GSA and MDBP.
- **Evaluation Utilities:** Scripts for visualizing results and computing pixel-level and image-level metrics.

> ⚠️ **Note:** Only limited sample images are included in this repository. The full source code, including implementations of the proposed algorithm and all traditional visual comparison methods, will be released after the manuscript is officially accepted to avoid unnecessary exposure of deployed software components.

---

## Environment Setup

- **Platform:** Visual Studio 2022  
- **Language:** C++  
- **Library:** OpenCV 3.4.16  
- **Build Type:** Release (x64)  

### Setup Guide

1. **Clone the repository**
   ```bash
   git clone https://github.com/SwaggyPinqi12/HSSTalgorithm.git
2. Open the Visual Studio solution file.
3. Configure OpenCV environment variables (```include```, ```lib```, and ```bin``` paths).
4. Build the project in *Release* mode.
5. Executables and results will appear under ```/bin```.