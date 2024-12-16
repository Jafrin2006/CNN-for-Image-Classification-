# CNN for Image Classification

TensorFlow/Keras is used in this project to create a Convolutional Neural Network (CNN) for picture classification. The model uses deep learning techniques to classify photos into predetermined categories.

## Features
- TensorFlow and Keras are used to create and train models.
- Conv2D, MaxPooling2D, and Dense are among the important CNN layers that are implemented.
- Incorporates batch normalization and dropout to enhance model performance and avoid overfitting.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- scikit-learn (optional, for performance evaluation)

Install dependencies with:
bash
pip install tensorflow numpy matplotlib scikit-learn


## Dataset
The project expects an image dataset organized in a directory structure:

- dataset/
  - train/
    - class1/
    - class2/
  - test/
    - class1/
    - class2/

Replace class1 and class2 with your actual class names. Update the dataset paths in the notebook accordingly.

## Model Architecture
The CNN is built using the following layers:
1. Convolutional layers with ReLU activation
2. MaxPooling layers for downsampling
3. Flatten layer to convert 2D features into a 1D vector
4. Dense layers for classification
5. Dropout layers to reduce overfitting
6. Batch normalization for faster convergence

## Usage
1. Go to the project directory after cloning the repository.
2. Get your dataset ready in the format needed.
3. Launch CNN for Image Classification.ipynb, a Jupyter Notebook file.
4. To load and preprocess the dataset, execute the cells one after the other.
   Construct and assemble the CNN model.
   Train the model and assess how well it performs.

## Test the accuracy
- I have made a function to test if my model is working properly and check the model with two demo images which is loaded in the single_prediction folder.
- The machine successfully recognise the correct picture as dog and cat.
## Results
Add your model's performance metrics here, such as:
- Accuracy: 86.5%

Include confusion matrices, loss graphs, or example predictions as applicable.

## Links
- [Dateset URl link](https://drive.google.com/drive/folders/1Ia7lGXcX-j3z8-iUza3Y8Z56n8D5rg12?usp=sharing)
- [My Medium Blog for this project](https://medium.com/@jislam_64383/building-an-image-classification-model-using-convolutional-neural-networks-cnns-1769c9ced457)

## Acknowledgments
- TensorFlow/Keras for offering a strong framework for deep learning.
- The dataset suppliers (if utilizing an external dataset, include details here).

