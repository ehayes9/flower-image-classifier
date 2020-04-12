import argparse
import numpy as np
import pandas as pd
import torch
import json
import PIL

from torchvision import datasets, transforms, models
from PIL import Image


def get_args():

	'''define arguments to receive from the command line
		'--' indicates the argument is an optional input
	'''
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser()

	parser.add_argument('image_file_name', 
                        type = str, 
                        help = 'Input the filepath & name of your image to be classified')

	parser.add_argument('gpu', 
                        type = bool, 
                        help = 'Toggle GPU mode on or off')
    
	parser.add_argument('--checkpoint_file', 
                        type = str, 
                        help = 'Input the name of the filename of your checkpoint to load')
    
	parser.add_argument('--top_k', 
                        type = int, 
                        help = 'Input the number of top classes to return')
    
	parser.add_argument('--category_mappings', 
                        type = str, 
                        help = 'Input the filepath & name of the json file with flower category mappings')
    
    
	args = parser.parse_args()
	return parser.parse_args()


def load_checkpoint(checkpoint_file,model):

	'''load model saved in checkpoint file to avoid having to re-build
	'''
	if type(checkpoint_file) == type(None):
		checkpoint = torch.load('checkpoint.pth')
	else: 
		checkpoint = torch.load(checkpoint_file)
    
	for param in model.parameters(): param.requires_grad = False
    
	model.class_to_idx = checkpoint['class_to_idx']
	model.classifier = checkpoint['classifier']
	model.load_state_dict(checkpoint['state_dict'])
    
	return model


def import_category_mappings(category_mappings):

	''' import json file that maps indices to flower categories
	'''
	if type(category_mappings) == type(None):
		category_mappings = 'cat_to_name.json'
	else:
		category_mappings = category_mappings
	with open(category_mappings, 'r') as f:
		category_mappings = json.load(f)
	return category_mappings
   

def process_image(image_file_name):
	''' scale, crop, and normalize a PIL image so it can be used as an input to the model,
		return an Numpy array
	'''
	#open image
	image = PIL.Image.open(image_file_name)

	#resize image     
	w,h = image.size
	if w > h:
		resize_ratio = w/h
		new_size = int(256 * resize_ratio),256
	elif h > w:
		resize_ratio = h/w
		new_size = 256,int(256 * resize_ratio)
	else:
		new_size = 256,256
	image = image.resize(new_size)
	
	#crop image
	crop_image = image.crop((new_size[0]//2-112,new_size[1]//2-112,new_size[0]//2+112,new_size[1]//2+112))
	
	#divide color channel by 255 to return floats between 0 & 1 
	img_array = np.array(crop_image)
	np_image = img_array/255
	
	# normalize color channels
	means = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	np_image = (np_image-means)/std
		
	# set color to first channel
	np_image = np_image.transpose(2, 0, 1)
	
	return np_image

def imshow(image, ax=None, title=None):
	"""Imshow for Tensor."""
	if ax is None:
		fig, ax = plt.subplots()
	
	# PyTorch tensors assume the color channel is the first dimension
	# but matplotlib assumes is the third dimension
	image = image.transpose((1, 2, 0))
	
	# Undo preprocessing
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image = std * image + mean
	
	# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
	image = np.clip(image, 0, 1)
	
	ax.imshow(image)
	
	return ax

def predict(image_path, model, gpu, top_k):
	''' Predict the class of an image using the trained model 
	'''
#     check whether top_k value has been input
	if type(top_k) == type(None):
		top_k = 5 
	else: 
		top_k = top_k
        
	if gpu == True:
		model.to("cpu")
	else:
		model.to("gpu")
        
	image = process_image(image_path)
	torch_image = torch.from_numpy(np.expand_dims(image, 
												  axis=0)).type(torch.FloatTensor).to("cpu")
	model.eval()
	output = model(torch_image)
	pr_output = torch.exp(output)
	probs,indices = pr_output.topk(top_k)
	probs = np.array(probs.detach())[0]
	indices = np.array(indices.detach())[0]

	# convert indices to actual category names
	index_to_class = {val: key for key, val in model.class_to_idx.items()}
	top_classes = [index_to_class[each] for each in indices]

	return probs,top_classes

if __name__ == "__main__":

	input = get_args()
	image_file_name = input.image_file_name
	checkpoint_file = input.checkpoint_file
	top_k = input.top_k
	category_mappings = input.category_mappings
	gpu = input.gpu   
	model = models.vgg16(pretrained=True)
	model.name = "vgg16"
    
#load model and check whether to use GPU mode    
	model = load_checkpoint(checkpoint_file,model)
 
#import category mappings
	category_mappings = import_category_mappings(category_mappings)
 
	probs, top_classes = predict(image_file_name, model, gpu, top_k)
    
	top_flowers = [category_mappings[each] for each in top_classes]

	print(f"Top flowers = {top_flowers}".format(top_flowers=top_flowers),
          f"\nProbabilities = {probs}".format(probs=probs))