import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import numpy as np
import json

def get_args():
	'''define arguments to receive from the command line
		'--' indicates the argument is an optional input
	'''
	parser = argparse.ArgumentParser()

	parser.add_argument('data_dir', 
		type = str, 
		help = 'Input the filepath of the data as a string')

	parser.add_argument('gpu', 
		type = bool, 
		help = 'Input True or False to toogle GPU mode')

	parser.add_argument('--arch', 
		type = str, 
		help = 'Input the name of the torchvision model as a string')    
    
	parser.add_argument('--learning_rate', 
		type = float, 
		help = 'Input the learning rate to use for backprop as a float with 3 decimal places')

	parser.add_argument('--epochs', 
		type = int, 
		help = 'Input the number of epochs as an integer')

	parser.add_argument('--hidden_units', 
		type = bool, 
		help = 'Input number of hidden units for model')
    
	args = parser.parse_args()
	return parser.parse_args()


def transform_data(data_dir,test_dir,valid_dir):
	'''for transforming and loading data
	'''
	train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

	test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),                                      
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

	train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
	test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
	eval_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

	trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
	evalloader = torch.utils.data.DataLoader(eval_data, batch_size=64)

	return train_data, test_data, eval_data, trainloader, testloader, evalloader

def load_model(arch):
	'''load pre-trained model
	'''
	#check for model input from command line
	if type(arch) == type(None):
		model = models.vgg16(pretrained=True)
		model.name = "vgg16"
		print("Pretrained model architecture = vgg16")
	else:
		exec("model = models.{}(pretrained=True)".format(arch))
		model.name = arch

	return model         

def create_classifier(model, hidden_units):
	'''create classifier with 2 fully connected layers using dropout, ReLu for gradient descent and softmax for output
	'''
	#check for hidden units input from command line
		if type(hidden_units) == type(None): 
				hidden_units = 4096 
				print("Number of hidden units = 4096")
		
		classifier = nn.Sequential(OrderedDict([
													('fc1', nn.Linear(25088, hidden_units, bias=True)),
													('relu1', nn.ReLU()),
													('dropout1', nn.Dropout(p=0.5)),
													('fc2', nn.Linear(hidden_units, 102, bias=True)),
													('output', nn.LogSoftmax(dim=1))
													]))
		model.classifier = classifier
		return model


def train_model(epochs,trainloader,steps,running_loss,print_every):
	'''train model, output training & validation loss, validation accuracy for each epoch
	'''	
	#check for epochs input from command line

	if type(epochs) == type(None):
		epochs = 5
		print("Number of epochs to train model = 5")
    
	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1

			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			output = model(inputs)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if steps % print_every == 0:
				model.eval()

				with torch.no_grad():
					eval_loss, accuracy = validation(model, evalloader, criterion)

				print("Epoch: {} , ".format(epoch+1),
                      "Training Loss: {:.3f} , ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} , ".format(eval_loss/len(evalloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(evalloader)))

				running_loss = 0
				model.train()
                
	return model
        
	print("\nTraining completed!")

def validation(model, testloader, criterion):
    '''validate model using test_loss & accuracy of test dataset
    '''
    test_loss = 0
    accuracy = 0
    
    model.eval()
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def save_checkpoint(model,file): 
	'''save model parameters to a checkpoint for ability to re-load tuned model 
	'''
	model.class_to_idx = train_data.class_to_idx

	checkpoint = {'arch': model.name,
              	'classifier': model.classifier,
              	'class_to_idx': model.class_to_idx,
              	'state_dict': model.state_dict()}

	torch.save(checkpoint, file)

def load_checkpoint(file,model):

    checkpoint = torch.load(file)
    
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model




if __name__ == "__main__":

	#get args from command line input
	input = get_args()

	arch = input.arch
	data_dir = input.data_dir
	learning_rate = input.learning_rate
	hidden_units = input.hidden_units
	epochs = input.epochs
	gpu = input.gpu
    
	#define directories for train, test & validation data
	
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# import json file to map flower names to indices
	with open('cat_to_name.json', 'r') as f:
		category_to_name = json.load(f)
    
    # load and transform data 
	train_data, test_data, eval_data, trainloader, testloader, evalloader = transform_data(train_dir,valid_dir,test_dir)

	model = load_model(arch)

    # freeze features parameters to speed up training

	for param in model.parameters():
		param.requires_grad = False  

	model = create_classifier(model,hidden_units=hidden_units)

# 	#set device to default = cuda for faster GPU processing
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	# only train the classifier parameters, feature parameters are frozen
	if type(learning_rate) == type(None):
		learning_rate = 0.001
		print("Learning rate = 0.001")
	else: 
		learning_rate = args.learning_rate
        
	optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()
	steps = 0
	running_loss = 0
	print_every = 30

	#train & evaluate model
	model = train_model(epochs,trainloader,steps,running_loss,print_every)

	# perform validation on the test set

	test_loss, equal = validation(model, testloader, criterion)
	accuracy = equal/len(testloader)*100
	print("Accuracy score for the test data is {:.3f} %".format(accuracy))

	save_checkpoint(model,'checkpoint.pth')

