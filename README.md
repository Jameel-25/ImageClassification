# Cat vs. Dog Image Classification

## Overview
This project is a deep learning-based image classification model designed to differentiate between images of cats and dogs. Using a convolutional neural network (CNN), the model is trained on a dataset of labeled images and is capable of accurately predicting the class of new, unseen images. The project is structured to guide you through the process of data preprocessing, model training, and evaluation.

## Project Steps

### 1. Collecting the Dataset
A diverse dataset containing labeled images of cats and dogs was gathered. The dataset was balanced to ensure equal representation of both classes, which is crucial for training an unbiased model.

### 2. Data Preprocessing
- **Image Resizing:** All images were resized to a consistent size of 100x100 pixels to standardize input dimensions.
- **Normalization:** Pixel values were normalized to a range of [0, 1] to help the model converge more efficiently.
- **Data Augmentation:** Techniques such as rotation and flipping were applied to the images to increase dataset variability and help the model generalize better.
- **Dataset Splitting:** The dataset was split into training, validation, and test sets to evaluate the model's performance effectively.

### 3. Model Training
- **Architecture:** A Convolutional Neural Network (CNN) was selected for its effectiveness in image classification tasks.
- **Training:** The model was trained on the training dataset using the Adam optimizer and binary cross-entropy loss function. Performance was monitored on the validation set to avoid overfitting.
- **Layers:** The model consists of multiple convolutional layers followed by max-pooling layers, dropout for regularization, and dense layers for final classification.

### 4. Model Evaluation
- **Metrics:** The model's performance was evaluated on the test set using accuracy as the primary metric.
- **Results:** The model achieved an accuracy of 77.85% on the test set, demonstrating its ability to correctly classify images of cats and dogs.

### 5. Testing
The model's generalization capabilities were assessed by evaluating its performance on the test set. The testing phase provided insights into how well the model can classify unseen data.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/Jameel-25/ImageClassification.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:
    ```bash
    python train_model.py
    ```
4. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```

## Results
The model demonstrates a strong ability to distinguish between images of cats and dogs, making it a valuable tool for automated image classification tasks in this domain.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
