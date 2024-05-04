# Vehicle Image Classifier

This repository contains the code for a Vehicle Image Classifier, which is a Flask web application combined with a convolutional neural network (CNN) to classify vehicle images. The model is trained to predict whether the vehicle in the image matches certain criteria (e.g., a specific type of vehicle), and the web app provides a user interface for uploading images and displaying predictions.

## Project Structure

The project is organized as follows:
- `app.py`: Flask application to handle web requests.
- `main.py`: Script to train the CNN model and generate performance metrics.
- `index.html`: Front-end HTML template for the web interface.
- `saved_model_and_weights/`: Directory containing the saved Keras model and weights.
- `figs/`: Directory for storing output figures such as loss and accuracy plots during training.
- `data/`: Directory containing training and testing image data.

## Installation

To set up and run this project, you will need Python 3.6+ and the following Python libraries installed:
- Flask
- NumPy
- PIL
- TensorFlow
- Matplotlib
- scikit-learn

You can install these packages with pip:

```bash
pip install flask numpy pillow tensorflow matplotlib scikit-learn
```
## Running the Application
Start the Flask app:
```bash
python app.py
```
This will run the web server on http://127.0.0.1:5000/.
Open a browser and navigate to http://127.0.0.1:5000/ to see the home page where you can upload vehicle images.

## Model Architecture
The CNN model (create_cnn3) is structured as follows:

Convolutional layers with increasing filter sizes and ReLU activations.
Batch normalization and MaxPooling for feature normalization and dimension reduction.
Dropout layers to prevent overfitting.
Fully connected layers at the end to output the prediction.
The model outputs a binary prediction: "Positive" if the probability is greater than 0.5, otherwise "Negative"

## Web Interface
The web interface allows users to upload images and view predictions. It includes JavaScript for handling the image upload and displaying the result without needing to refresh the page.
