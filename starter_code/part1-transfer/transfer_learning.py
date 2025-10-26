# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

#<<<YOUR CODE HERE>>>
data_dir = r"C:\Users\DELL\Downloads\cd12528-small-data-project-starter-main\cd12528-small-data-project-starter-main\starter_code\part1-transfer\imagedata-50"

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}



#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

#<<<YOUR CODE HERE>>>
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val', 'test']
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for x in ['train', 'val', 'test']
}

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

class_names = image_datasets['train'].classes
print("Classes:", class_names)
#hint, create a variable that contains the class_names. You can get them from the ImageFolder



# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#<<<YOUR CODE HERE>>>
from torchvision.models import VGG16_Weights
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)


# Freeze all layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace top classifier
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(class_names))
# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>
num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# When you have all the parameters in place, uncomment these to use the functions imported above
#def main():
#   trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
#   test_model(test_loader, trained_model, class_names)

#if __name__ == '__main__':
#    main()
#    print("done")

def main():
    trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
    test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
    main()
    print("Training and testing Done✅")