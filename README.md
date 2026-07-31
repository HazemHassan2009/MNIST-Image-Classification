# MNIST Digit Classification using ResNet50 Transfer Learning

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project demonstrates how to classify handwritten digits from the MNIST dataset using a pre-trained ResNet50 model through transfer learning. The approach involves leveraging the powerful feature extraction capabilities of ResNet50, a deep convolutional neural network, and fine-tuning it with custom classification layers for the specific task of MNIST digit recognition. The notebook covers data loading, preprocessing, model definition, training, and evaluation.

## Project Structure
- `mnist_resnet50.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model definition, training, and evaluation.

## Installation
To run this notebook, you need to have Python and several libraries installed. It is recommended to use a virtual environment.

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install tensorflow matplotlib scikit-learn seaborn numpy
    ```

## Usage
To execute the code and reproduce the results, simply open and run the `mnist_resnet50.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, or Google Colab).

1.  **Open the notebook:**
    ```bash
    jupyter notebook mnist_resnet50.ipynb
    ```
    or upload it to Google Colab.

2.  **Run all cells:** Execute each cell sequentially from top to bottom. The notebook will:
    -   Load the MNIST dataset.
    -   Preprocess the images (normalization, reshaping to RGB-like, resizing to 32x32, and ResNet-specific preprocessing).
    -   Define a transfer learning model using a pre-trained ResNet50 base.
    -   Train the model.
    -   Evaluate the model's performance with a classification report and confusion matrix.

## Model Details
### Architecture
The model uses a pre-trained ResNet50 as a feature extractor. The `include_top` argument is set to `False` to remove the original ImageNet classification head. Custom layers are then added:
-   A `GlobalAveragePooling2D` layer to reduce the feature maps.
-   A `Dense` layer with 10 units (for 10 MNIST classes) and `softmax` activation for classification.

### Transfer Learning Strategy
The layers of the base ResNet50 model are initially frozen (`layer.trainable = False`) to retain the learned ImageNet features. This prevents the large pre-trained model from overfitting on the smaller MNIST dataset and speeds up training.

### Training Configuration
-   **Optimizer**: Adam
-   **Loss Function**: Sparse Categorical Crossentropy (suitable for integer labels)
-   **Metrics**: Accuracy
-   **Epochs**: 5 (can be adjusted)
-   **Batch Size**: 32

## Results
After training for 5 epochs, the model achieved the following performance on the test set:
-   **Test Loss**: ~0.3747
-   **Test Accuracy**: ~0.8849

The classification report and confusion matrix provide further insights into the per-class performance and common misclassifications.

### Classification Report
```
              precision    recall  f1-score   support

           0       0.91      0.96      0.93       980
           1       0.92      0.99      0.96      1135
           2       0.78      0.93      0.85      1032
           3       0.86      0.86      0.86      1010
           4       0.94      0.91      0.93       982
           5       0.89      0.83      0.86       892
           6       0.88      0.93      0.91       958
           7       0.94      0.86      0.90      1028
           8       0.95      0.64      0.77       974
           9       0.83      0.91      0.87      1009

    accuracy                           0.88     10000
   macro avg       0.89      0.88      0.88     10000
weighted avg       0.89      0.88      0.88     10000
```

### Confusion Matrix
(Visualized as a heatmap in the notebook)

## Contributing
Feel free to fork the repository, open issues, or submit pull requests. Any contributions are welcome!

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
