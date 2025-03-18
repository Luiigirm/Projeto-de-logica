# Projeto-de-logica
# O seguinte dataset foi descompactado dentro da pasta "dog_cat": 
# https://www.kaggle.com/datasets/tongpython/cat-and-dog

import os
import copy
import joblib
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
path_save_models = 'models'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')
Device: cuda
Resnet18 - dog_cat
class CustomDataset(Dataset):
    def _init_(self, path_images, labels, transform, transform_augmentation=None):
        self.path_images = path_images
        self.labels = labels
        self.transform = transform
        self.augmentation = transform_augmentation
    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        img = Image.open(self.path_images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.augmentation:
            img = self.augmentation(img)
        else:
            img = self.transform(img)

        return img, torch.tensor(np.array([label]), dtype=torch.float32)

def get_image_path(path):
    X = []
    y = []
    for cls, folder in enumerate(os.listdir(path)):
        full_path = f'{path}/{folder}'
        for filename in os.listdir(full_path):
            if filename.endswith('jpg') or filename.endswith('png'):
                X.append(f'{full_path}/{filename}')
                y.append(cls)
    return X, y

path_train = 'data/dog_cat/training_set/training_set'
path_valid = 'data/dog_cat/test_set/test_set'

X_train, y_train = get_image_path(path_train)
X_valid, y_valid = get_image_path(path_valid)

model = models.resnet18(weights='IMAGENET1K_V1')
model
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1, bias=True)
)
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

transform_default = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

transform_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
dataset_train = CustomDataset(X_train, y_train, transform_default, transform_augmentation)
dataset_valid = CustomDataset(X_valid, y_valid, transform_default)

train_loader = torch.utils.data.DataLoader(
                    dataset_train, batch_size=128,
                    shuffle=True, num_workers=4)

valid_loader = torch.utils.data.DataLoader(
                dataset_valid, batch_size=128,
                shuffle=False, num_workers=4)
                
log_name = ''
dataset_name = 'dog_cat'
total_epoch = 10


if log_name != '':
    writer = SummaryWriter(f'runs/{log_name}')  # 'runs/50'
else:
    writer = SummaryWriter()

results = []
# acc, loss, epoch
best_acc = [0., 0., 0]

for epoch in range(total_epoch):
    torch.cuda.empty_cache()
    model.train()
    running_loss = 0.
    running_accuracy = 0.
    running_p = []
    running_y = []
    for data in tqdm(train_loader):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(torch.sigmoid(outputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        outputs = (torch.sigmoid(outputs) > 0.5).float()
        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach()
        running_accuracy += accuracy_score(labels, outputs)
    writer.add_scalar(f'{dataset_name}-Accuracy/train', running_accuracy/len(train_loader), epoch)
    writer.add_scalar(f'{dataset_name}-Loss/train', running_loss/len(train_loader), epoch)
    print('Epoch {}; Loss {}; Accuracy {}'.format(epoch, running_loss/len(train_loader), running_accuracy/len(train_loader)))

    if epoch % 1 == 0:
        model.eval()
        running_accuracy_valid = 0.
        running_loss_valid = 0.
        running_p2 = []
        running_y2 = []
        for data in tqdm(valid_loader):
            inputs_valid, labels_valid = data
            
            inputs_valid = inputs_valid.to(device)
            labels_valid = labels_valid.to(device)

            outputs_valid = model(inputs_valid)
            loss = criterion(torch.sigmoid(outputs_valid), labels_valid)
            running_loss_valid += loss.item()

            outputs_valid = (torch.sigmoid(outputs_valid) > 0.5).float()
            outputs_valid = outputs_valid.cpu().detach()
            labels = labels_valid.cpu().detach()
            running_accuracy_valid += accuracy_score(labels, outputs_valid)
        writer.add_scalar(f'{dataset_name}-Accuracy/valid', running_accuracy_valid/len(valid_loader), epoch)
        writer.add_scalar(f'{dataset_name}-Loss/valid', running_loss_valid/len(valid_loader), epoch)
        print(f'Loss valid {running_loss_valid/len(valid_loader)}; Accuracy valid:{running_accuracy_valid/len(valid_loader)}')
        if running_accuracy_valid/len(valid_loader) > best_acc[0]:
            print('### New best model! ###')
            best_acc = [running_accuracy_valid/len(valid_loader), running_loss_valid/len(valid_loader), epoch]
            best_model = copy.deepcopy(model)
    writer.flush()
writer.flush()
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 0; Loss 0.2149312026680462; Accuracy 0.9074397573038878
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.95006390902563; Accuracy valid:0.72119140625

  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 1; Loss 0.11456036029590501; Accuracy 0.9523108609385784
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.13747059274464846; Accuracy valid:0.9555806280339806

  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 2; Loss 0.10314769989677838; Accuracy 0.960229396710375
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.1494202297180891; Accuracy valid:0.9442316444174758
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 3; Loss 0.08791177258605049; Accuracy 0.9658978174603174
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.20482293085660785; Accuracy valid:0.9148020327669903
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 4; Loss 0.06359073975019985; Accuracy 0.9773424919484702
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.16835305001586676; Accuracy valid:0.9393488319174758
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 5; Loss 0.0714710262559709; Accuracy 0.9721341586151369
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.1758855665102601; Accuracy valid:0.9369216474514563
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 6; Loss 0.05688929081790977; Accuracy 0.9777145157579941
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.16242208401672542; Accuracy valid:0.937158677184466
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 7; Loss 0.053975277402926056; Accuracy 0.9789366229583621
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.14853878458961844; Accuracy valid:0.9415389866504854
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 8; Loss 0.05247705883627373; Accuracy 0.9796285512997469
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.1560303580481559; Accuracy valid:0.9499725045509708
  0%|          | 0/63 [00:00<?, ?it/s]
Epoch 9; Loss 0.058551691295135586; Accuracy 0.9763684006211181
  0%|          | 0/16 [00:00<?, ?it/s]
Loss valid 0.1718963256571442; Accuracy valid:0.9515700849514563

pathlib.Path(path_save_models).mkdir(exist_ok=True, parents=True)
torch.save(best_model, f'{path_save_models}/dog_cat.pth')

best_model = torch.load(f'{path_save_models}/dog_cat.pth')

transform_default = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
