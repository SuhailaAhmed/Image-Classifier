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
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

# TODO: Define your transforms for the training, validation, and testing sets

# transform training data
train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# transform validation data
valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# transform testing data
test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

# TODO: Load the datasets with ImageFolder
training_data = datasets.ImageFolder(train_dir, transform = train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
testing_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
training_loader = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle = True)
testing_loader = torch.utils.data.DataLoader(testing_data, batch_size = 64, shuffle = True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network

# Load pretrained vgg16 Model
model = models.vgg16(pretrained=True)

# freeeze part of pretrained vgg16 Model
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('input1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('input2', nn.Linear(4096, len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# use model on GPU
if torch.cuda.is_available():
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.classifier

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

epochs = 10
s = 0
p = 20

for epoch in range(epochs):
    training_loss = 0
    training_accuracy = 0
    model.train()
    
    for ins, labels in training_loader:
        training_loss = 0
        training_accuracy = 0
        model.train()
        s += 1
        ins = ins.to('cuda')
        labels = labels.to('cuda')
        
        optimizer.zero_grad()
        
        outs = model.forward(ins)
        # calculate loss
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        #  calculate accuracy
        exp = torch.exp(outs)
        top_exp, top_class = exp.topk(1,dim=1)
        m = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        acc = m.mean()
        
        training_loss += loss.item()
        training_accuracy += acc.item()
        
        if s%p == 0:
            validation_loss = 0
            validation_accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for val_ins,val_labels in validation_loader:
                    val_ins = val_ins.to('cuda')
                    val_labels = val_labels.to('cuda')
                    
                    val_outs = model.forward(val_ins)
                    # calculate loss
                    val_loss = criterion(val_outs,val_labels)
                    #  calculate accuracy
                    val_exp = torch.exp(val_outs)
                    val_top_ps, val_top_class = val_exp.topk(1,dim=1)
                    val_m = (val_top_class == val_labels.view(*val_top_class.shape)).type(torch.FloatTensor)
                    val_acc = val_m.mean()

                    validation_loss += val_loss.item()
                    validation_accuracy += val_acc.item()
                
                    
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"Training Loss: {training_loss/p:.3f}  "
                  f"Training Accuracy: {training_accuracy/p*100:.3f}%  "
                  f"Validation Loss: {validation_loss/len(validation_loader):.3f}  "
                  f"Validation Accuracy: {validation_accuracy/len(validation_loader)*100:.3f}%")

# TODO: Do validation on the test set
testing_accuracy = 0

with torch.no_grad():
    for test_ins,test_labels in testing_loader:
        model.eval()

        test_ins = test_ins.to('cuda')
        test_labels = test_labels.to('cuda')

        test_outs = model.forward(test_ins)
        test_exp = torch.exp(test_outs)

        test_top_exp,test_top_class = test_exp.topk(1,dim=1)
        test_m = (test_top_class == test_labels.view(*test_top_class.shape)).type(torch.FloatTensor)
        test_acc = test_m.mean()
        testing_accuracy += test_acc
    
print(f'Testing Accuracy: {testing_accuracy/len(testing_loader)*100:.2f}%')

# TODO: Save the checkpoint 
model.class_to_idx = training_data.class_to_idx
torch.save({'structure': 'vgg16',
            'input': 25088,
            'output': len(cat_to_name),
            'hidden':4096,
            'learning_rate': 0.001,
            'epochs': epochs,
            'classifier': model.classifier,
            'optimizer_dict': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'model.class_to_idx': model.class_to_idx},
            'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model

def loading_the_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
