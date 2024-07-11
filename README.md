# Glacier Image Segmentation and Error Analysis

This repository contains code for segmenting glacier images, applying selective mode filtering, and calculating error metrics to evaluate the quality of the segmentation. The process involves patchifying images, predicting segmentation masks using a trained model, applying selective mode filtering to the predictions, and comparing the results to ground truth masks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare the Dataset](#prepare-the-dataset)
  - [Run the Segmentation and Filtering](#run-the-segmentation-and-filtering)
- [Results](#results)
- [License](#license)

## Introduction

This project aims to segment glacier images into different classes using a deep learning model and evaluate the performance of the segmentation. The process includes:
1. Patchifying the input images.
2. Predicting segmentation masks using a pre-trained model.
3. Applying selective mode filtering to the predicted masks to reduce noise.
4. Calculating error metrics by comparing the predicted masks with ground truth masks.

## Features

- Patchify large images into smaller patches for easier processing.
- Predict segmentation masks using a pre-trained deep learning model.
- Apply selective mode filtering to enhance the segmentation results.
- Calculate error metrics (Total Pixel Error, Mean Squared Error, Root Mean Squared Error) to evaluate the quality of the segmentation.

## Installation

To use this code, you'll need to install the required dependencies. You can install them using `pip`:

```sh
pip install numpy opencv-python tensorflow keras pillow matplotlib scikit-learn segmentation-models patchify
```

## Usage

### Prepare the Dataset

1. **Image Directory Structure:**
   Ensure that your dataset is structured as follows:
   ```
   root_directory/
   ├── images/
   │   ├── image1.tif
   │   ├── image2.tif
   │   └── ...
   ├── masks/
   │   ├── mask1.tif
   │   ├── mask2.tif
   │   └── ...
   ```

2. **Trained Model:**
   Place your trained model file (`.keras` format) in a known location. Update the `model_path` variable in the code with the path to your model.

### Run the Segmentation and Filtering

Run the main script to perform segmentation, filtering, and error calculation:

```sh
python main_script.py
```

The script will:
1. Patchify each image.
2. Use the model to predict segmentation masks for each patch.
3. Unpatchify the predictions to reconstruct the full-size segmentation mask.
4. Apply selective mode filtering to reduce noise in the predicted masks.
5. Compare the filtered and unfiltered predictions with the ground truth masks and calculate error metrics.
6. Save the results and images to specified directories.

## Results

The script saves the following results:
- **Unpatchified Predictions:** Saved in `patchify/unpatchified/` directory.
- **Filtered Predictions:** Saved in `patchify/filtered/` directory.
- **Comparison Images:** Saved in `patchify/pictures/` directory.
- **Error Metrics:** Saved in `results.txt` file in the root directory.

Example of results stored in `results.txt`:
```
**6x5 Fixed Double Pixel Mode Filter:**
Number of Changed Border Pixels:  1034
Percent of Border Pixels Changed:  0.5

Total Pixel Error: 123456
Mean Squared Error (MSE): 78.9
Root Mean Squared Error (RMSE): 8.88
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to open issues or pull requests for any improvements or questions related to this project. Contributions are welcome!
