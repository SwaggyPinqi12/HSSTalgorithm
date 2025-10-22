# Deep Learning models for Fabric Stain Detection

This folder contains the comparison deep learning models implementations for fabric stain detection, used for comparison with the proposed HSST algorithm.

The implementations leverage the **[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)** open-source project and support a variety of semantic segmentation models including:

- Unet++
- DeeplabV3
- UperNet
- SegFormer
- Other segmentation models supported by segmentation-models-pytorch

---

## Environment Setup

- **Platform:** WSL2 (Ubuntu 22.04)  
- **Python:** 3.10.16  
- **CUDA:** 12.1  
- **PyTorch:** 2.1.2+cu121  
- **segmentation-models-pytorch:** 0.5.1.dev0  
- **Pillow:** 11.0.0  

### Installation

```bash
# Clone the main repository
git clone https://github.com/SwaggyPinqi12/HSSTalgorithm.git
cd HSSTalgorithm/dl_baselines

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch==2.1.2+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch==0.5.1.dev0 pillow==11.0.0
```

## Usage

### Training
Train a model using ```train.py```. Example:

```bash
python train.py -b 32 -e 1000
```

### Inference
Predict on a folder of images using ```predict_folder.py```:

```bash
python predict_folder.py
```

- Note: Only a subset of dataset samples is included in this repository. Full dataset and pre-trained models will be released after the manuscript is formally accepted. 
