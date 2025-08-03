# Concrete-Crack-Detection

## Overview

This project implements a crack recognition system using a Convolutional Neural Network (CNN) built with PyTorch. The system is designed to detect cracks in images, specifically from the **Cracks Dataset** available on Mendeley. The dataset contains images categorized into:

- **Positive**: Images with cracks  
- **Negative**: Images without cracks

---

## Dataset

The dataset used in this project is the **Cracks Dataset**. It includes:

- **20,000 Crack Images** (`Positive/` directory)  
- **20,000 No-Crack Images** (`Negative/` directory)

These images are used for training and evaluating the model to distinguish between surfaces with and without cracks.

---

## Requirements

To run the code in the provided Jupyter notebook (`Crack-recognition.ipynb`), you need the following dependencies:

- Python 3.11.5  
- PyTorch  
- Torchvision  
- NumPy  
- OpenCV (cv2)  
- Matplotlib  
- PIL (Pillow)  
- Other standard Python libraries: `os`, `time`, `copy`, `random`, `shutil`, `re`


## Project Structure

Crack-recognition.ipynb      # Main Jupyter notebook
Positive/                    # Directory with crack images
Negative/                    # Directory with no-crack images
real_images/                 # Sample test images (e.g., road_surface_crack3.jpg)
pretrained_net_G.pth         # Trained model weights
base_model.pth               # Another model checkpoint

## Key Components

### 🔹 Data Loading and Visualization
- Loads images from the `Positive` and `Negative` directories.
- Prints the number of images in each category.
- Visualizes a sample test image using the `predict_on_crops` function with Matplotlib.

### 🔹 Model
- Uses a PyTorch-based CNN model (`base_model`) for crack detection.
- The trained model is saved as `pretrained_net_G.pth`.

### 🔹 Prediction
- `predict_on_crops(image_path, crop_h, crop_w)` processes the input image by splitting it into 128x128 pixel crops.
- Each crop is evaluated for crack presence.
- The result is visualized using OpenCV and converted to RGB for display.
## Usage

### ✅ Setup
- Ensure dataset images are placed in the `Positive/` and `Negative/` directories.
- Place test images in the `real_images/` directory.

### ✅ Running the Notebook
- Open `Crack-recognition.ipynb` in Jupyter.
- Run all cells sequentially to:
  - Load data  
  - Train the model  
  - Predict cracks in test images
## Saving the Trained Model

```python
torch.save(base_model.state_dict(), 'pretrained_net_G.pth')
print("Model saved to pretrained_net_G.pth")


