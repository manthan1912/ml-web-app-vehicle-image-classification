# Question 1
1. Obtained the vehicle detection image dataset from Kaggle for binary image classification.

2. Performed Image Processing and splitted the dataset into 'train', 'validation' and 'test' data using Keras ImageDataGenerator.
3. Applied data augmentation techniques to improve the model's learning and generalize the model.
4. Developed a custom Convolutional Neural Network (CNN) architecture for vehicle image classification.

5. Implemented three CNN architectures, including transfer learning with VGG16 for image classification.

6. Applied Dropout and Batch Normalization layers to prevent overfitting and improve model generalization.
7. Adjusted the learning rate of the Adam optimizer for better model convergence by fine-tuning it.
8. Implemented training and evaluation of the CNN model using train, validation, and test datasets.
9. Generated and saved model predictions on the test dataset.

10. Visualized training and validation loss and accuracy during the training process.

11. Constructed a confusion matrix to evaluate the classification performance of the model.

# Question 2
Dataset description:

The dataset is obtained from kaggle. It contains images for both classes - 'vehicles' and 'non-vehicles'. It is a balanced dataset with 8792 RGB images of class 1 (vehicles) and 8968 RBG images of class 0 (non-vehicles).
Since, I'm working with image dataset, computing AUC values for each measurement is not possible.

| Class        | Number of images |
|:-------------|:----------------:|
| Vehicles     |       8792       |
| Non-vehicles |       8968       |
| Total        |      17760       |

Each image is in the shape of (64x64x3) where 3 is the number of channels i.e. RGB in this case. The height and width of the images are 64 and 64 respectively.

Link to the dataset:

[1] https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set

# Question 3:

1. Kaggle 'data' folder had two sub-folders called 'non-vehicles' and 'vehicles'. This is a binary image classification problem. The classes are 1 for 'vehicle' and 0 for 'non-vehicle'.
I randomly picked equal set of images from both classes and put it into 'test' folder. There are total of 1775 images in test folder.
2. Used ImageDataGenerator to preprocess the train, validation and test data. Rescaling each data by dividing each value with 255.0 is a common operation for all the three datasets.
Also, all the images are reshaped into (64,64,3). Used batch_size of 32. 'class_mode' = 'binary' tells the generator to generate the labels in binary fashion.
3. Performed data augmentation only on the train images using rotation_range, width_shift_range,
height_shift_range, shear_range, zoom_range, horizontal_flip, fill_mode parameters. 
The real world data is much complex and can contain images in abnormal form such as skewed or zoomed in/out etc and since the training data does not take into account of these variations, we use data augmentation techniques to include such variations to our training images.
Data augmentation is very helpful when training the model with variations of the original image. This helps model to learn more complex information and improves the accuracy.

4,5,6,7,8.  <br>
I've developed two custom CNN models. First cnn_model (create_cnn1()) is a basic CNN with 4 convolutional layers, batch normalization, max-pooling, and fully connected layers with dropout. There are total 7 layers in this model.
The second custom CNN model (create_cnn2()) is an improved version of the basic model. It has same amount of convolution layers but different dropout, and batch normalization. This architecture also includes Dropout layers after each MaxPooling2D layer to help prevent overfitting.
Next, I tried the concept of transfer learning and fine-tuning of the pre-trained model. I used pre-trained VGG16 model (create_vgg16()) to train the images. The VGG16 base model includes 13 convolutional layers and 5 max-pooling layers.
The pre-trained VGG16 model is fine-tuned by adding a Flatten layer and three Dense layers, with Dropout layers in between.
Every model is trained with custom learning rate of 0.00001 for 'Adam' optimiser and epochs = 30. Each model is trained using 'train_gen' data and evaluated on 'val_gen' data. Finally, predictions are made on 'test_gen' data.

9. Made predictions on 'test_gen' data and saved the prediction results of each model as a numpy .npy file and can be found inside the 'outputs' folder.
10. Plotted and saved two graphs of each model. One is for training loss v/s. validation loss and other is for training accuracy v/s. validation accuracy. These plots can be found inside the 'figs' folder. 
11. Constructed the confusion matrix for evaluating the performance of each model. Confusion matrix of each model is saved and can be found in 'figs' folder.

#### Some key Insights:
* After running each model, the following accuracy and loss was obtained.

| Model            | Accuracy | Loss   |
|:-----------------|:--------:|:-------|
| CNN_1 (Basic)    |  0.9944  | 0.0234 |
| VGG16            |  0.9881  | 0.0303 |
| CNN_2 (improved) |  0.9987  | 0.0024 |

Here, we can see that the model which had Dropout layers after each MaxPooling2D layer performed better than the other models.
Surprisingly, the pre-trained model VGG16, which is already trained on a huge dataset (imagenet),
is performing lower than other models. Not so good fine-tuning of the top layers can be one of the reason.
