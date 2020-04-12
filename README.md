### Flower Image Classifier

Developed a deep neural network to recognize different images of flowers using Pytorch as part of Udacity's Machine Learning Nanodegree.

### Part I - Deep Learning Model 
Leveraged features of pre-trained models from torchvision to build the model. Created a feed-forward classifier with two fully connected layers using ReLU, dropout and the softmax function. 

By adjusting hyperparameters, achieved a validation accuracy of 89\%. 

Defined functions to perform pre-processing of the data for training, testing and evaluation. Also wrote functions to resize, crop and convert images to be accepted as inputs for the model for predicting. 

### Part II - Command Line Application

Created two scripts, train.py and predict.py that enable a user to build the model from the command line.

Train.py gives users the ability to specify model parameters by providing inputs such as model architecture, number of epochs, learning rate etc. The script outputs model validation loss and accuracy, and saves a checkpoint.

The predict.py script loads a checkpoint, accepts an image and returns the predicted classifier for the image. 

### Building and using the model

**Train.py**

```Required inputs: 	
- data_dir  (filepath of data to train model)
- gpu (T/F)
```

```Optional inputs:
--arch
--learning_rate
--epochs
--hidden_units
```

**Predict.py**

```Required inputs: 	
- image_file_name (file path to image to process & predict)
- gpu (T/F)
```

```Optional inputs:
--checkpoint_file (model to load)
--top_k (number of top predictions to return)
--category_mappings (json file with mappings from categories ot flower names)
 ```	



