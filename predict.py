import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from collections import OrderedDict
import json
import PIL
import seaborn as sns
from PIL import Image
import time

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda')
    model.eval()
    image = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([image])).float()

    with torch.no_grad():
        probs = model.forward(img.cuda())
        
    prob = torch.exp(probs).data
    
    return prob.topk(topk)

img = plt.imread(image_path)
img = img / 255.0

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Predict the top 5 classes for the image
probs, classes = predict(image_path, model)
print(probs, classes)
probs = probs.squeeze().cpu().numpy()
classes = [cat_to_name[str(cls)] for cls in classes.squeeze().cpu().numpy()]

# Plot the image and the top 5 classes as a bar chart
fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
ax1.imshow(img)
ax1.axis('off')
ax1.set_title(cat_to_name[str(69)])

y_pos = np.arange(len(classes))
ax2.barh(y_pos, probs, align='center')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(classes)
ax2.invert_yaxis()
ax2.set_title('Top 5 Predictions')

plt.show()