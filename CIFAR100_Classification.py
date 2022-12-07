import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim import lr_scheduler

import time
import copy
        
class CIFAR100_Classifier:
    '''Pipeline'''
    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 learning_rate,
                 num_epochs,
                 batch_size,
                 train_transform=None,
                 test_transform=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
        
        self.model = model.to(self.device)
        self.criterion = loss
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.train_transform = train_transform
        self.test_transform = test_transform
               
    def loadData(self):
        
        if not self.train_transform:
            print(f'No transformations for Training Images. Using original images...\n')
            self.train_transform = transforms.ToTensor()
            
        if not self.test_transform:
            print(f'No transformations for Training Images. Using original images...\n')
            self.test_transform = transforms.ToTensor()
        
        train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                      train=True,
                                                      download=True,
                                                      transform=self.train_transform)
        
        test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                     train=False,
                                                     download=True,
                                                     transform = self.test_transform)
        
        val_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=self.test_transform)
                
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.batch_size,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        shuffle=True)
        
        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=self.batch_size*2,
                                                      num_workers=4,
                                                      pin_memory=True,
                                                      shuffle=False)
        
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.batch_size*2,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       shuffle=False)
        
        print(f'\nTrain set size: {len(self.train_loader.dataset)}')
        print(f'Val set size: {len(self.val_loader.dataset)}')
        print(f'Test set size: {len(self.test_loader.dataset)}')
        
        self.data_loaders = {'train': self.train_loader,
                             'val': self.val_loader,
                             'test': self.test_loader}
        
        self.classes = train_dataset.classes
        print(f'\nApplied Transformations and loaded data\n')
        
        
    def train(self, grad_clip):
        
        # history = []       
        print(f'batch size: {self.batch_size}, # batches(steps per epoch): {len(self.train_loader)}\n')
        
        scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                             max_lr=self.learning_rate,
                                             epochs=self.num_epochs, 
                                             steps_per_epoch=len(self.train_loader))
        
        print(f'Training...\n')
        since = time.time()
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc =0.0

        for epoch in range(self.num_epochs):

            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-'*10)
            
            '''Training Phase'''
            for phase in ['train', 'val']:
                if phase =='train':
                    self.model.train()
                    self.train_losses = []
                    self.lrs = []
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                for i, (images, labels) in enumerate(self.data_loaders[phase]):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    '''Forward pass'''
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = self.model(images)
                        _, preds = torch.max(outputs, dim=1)
                        loss = self.criterion(outputs, labels)
                        
                        if phase == 'train':
                            
                            self.train_losses.append(loss)
                            
                            '''Grdient clipping'''
                            if grad_clip:
                                nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)
                            
                            '''Backward pass'''
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            
                            for param_group in self.optimizer.param_groups:
                                self.lrs.append(param_group['lr'])
                            
                            scheduler.step()
                    
                    #statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # if phase == 'train':
                #     scheduler.step()
                
                epoch_loss = running_loss / len(self.data_loaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.data_loaders[phase].dataset)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
            print()
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f} at epoch:{best_epoch+1}\n')
    
        self.model.load_state_dict(best_model_wts)
        
    def test(self):

        print(f'Testing...\n')

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(100)]
            n_class_samples = [0 for i in range(100)]

            for b, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            for i in range(100):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {self.classes[i]}: {acc} %')
            print()