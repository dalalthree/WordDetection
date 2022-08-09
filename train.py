#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

from IPython.display import display
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from numpy.random import choice

import random
import time
from collections import deque

import sys
import os

print('python: ' + str(sys.version))
print('torch: ' + str(torch.__version__))
print('torchvision: ' + str(torchvision.__version__))

#Datasets and Loading

class OmniglotTrainingSet(Dataset):
    def __init__(self, path, transform = None):
        super(OmniglotTrainingSet, self).__init__()
        self.seed = 1
        np.random.seed(self.seed)
        self.transform = transform
        self.images, self.class_count = self.loadData(path)
        
    def loadData(self, path):
        images = {} #stores all images loaded with identical character under one key
        idCount = 0 #number of different character types
        for _ in range(4): #allows possibility of all 4 rotations even if its not guarenteed
            for primary in os.listdir(path): #language character comes from
                for characterPath in os.listdir(os.path.join(path, primary)): #character type number
                    images[idCount] = []
                    for individualTest in os.listdir(os.path.join(path, primary, characterPath)): #each character image
                        f_path = os.path.join(path, primary, characterPath, individualTest)
                        #images[idCount].append(Image.open(f_path))
                        images[idCount].append(f_path)
                    idCount += 1
        return images, idCount
    
    def __getitem__(self, i):
        label, imgA, imgB = None, None, None
        
        # i ensures that theres a mix of same and different sets
        
        #same class
        if i % 2 == 1:
            label = torch.from_numpy(np.array([1.00], dtype=np.float32))
            idClass = random.randint(0, self.class_count - 1)
            imgA = random.choice(self.images[idClass])
            imgB = random.choice(self.images[idClass])

        #different class
        else:
            label = torch.from_numpy(np.array([0.00], dtype=np.float32))
            idClassA, idClassB = random.randint(0, self.class_count - 1), random.randint(0, self.class_count - 1)
            while idClassA == idClassB: #prevents same class
                idClassB = random.randint(0, self.class_count - 1)
            imgA = random.choice(self.images[idClassA])
            imgB = random.choice(self.images[idClassB])
        
        imgA = Image.open(imgA).convert('L')
        imgB = Image.open(imgB).convert('L')
        
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        
        return imgA, imgB, label
    
    def __len__(self):
        return  21000000 #5250000

class OmniglotTestingSet(Dataset):
    def __init__(self, path, numTests, way, transform=None):
        super(OmniglotTestingSet, self).__init__()
        self.seed = 2
        np.random.seed(self.seed)
        self.images, self.class_count = self.loadData(path)
        self.num_tests = numTests
        self.way = way
        self.classA, self.imgA = None, None #allows for n-way learning
        self.transform = transform
        
    def loadData(self, path):
        images = {} #stores all images loaded with identical character under one key
        idCount = 0 #number of different character types
        for primary in os.listdir(path): #language character comes from
            for characterPath in os.listdir(os.path.join(path, primary)): #character type number
                images[idCount] = []
                for individualTest in os.listdir(os.path.join(path, primary, characterPath)): #each character image
                    f_path = os.path.join(path, primary, characterPath, individualTest)
                    #images[idCount].append(Image.open(f_path))
                    images[idCount].append(f_path)
                idCount += 1
        return images, idCount
    
    def __getitem__(self, i):
        imgA, imgB = None, None
        
        # i ensures that theres a mix of same and different sets
        
        #same class
        if i % self.way == 0:
            self.classA = random.randint(0, self.class_count - 1)
            self.imgA = random.choice(self.images[self.classA])
            imgB = random.choice(self.images[self.classA])

        #different class
        else:
            idClassB = random.randint(0, self.class_count - 1) #set B every time but A only n-way times
            while self.classA == idClassB: #prevents same class
                idClassB = random.randint(0, self.class_count - 1)
            imgB = random.choice(self.images[idClassB])
        
        imgA = Image.open(self.imgA).convert('L')
        imgB = Image.open(imgB).convert('L')
        
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        
        return imgA, imgB 

    def __len__(self):
        return self.num_tests * self.way

 #Training Paramaters
iterations = 7000
batchSize = 100
learningRate = 0.001 #0.00006
way = 20
modelPath = 'data/saved/models'

data_transforms = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.ToTensor()
])

#Data preperations

trainSet = OmniglotTrainingSet('./docs/omniglot/images_background/images_background/', transform=data_transforms)
testSet = OmniglotTestingSet('./docs/omniglot/images_evaluation/images_evaluation/', transform=transforms.ToTensor(), numTests = 400, way = 20)

testLoader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=0)
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=False, num_workers=0)


#Check GPU
device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
print((str(device) + " - " + str(torch.cuda.get_device_name(torch.cuda.current_device()))))


#Siamese Model
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)
        
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #return self.sigmoid(out)
        return out
    
if __name__ == '__main__':
    net = Siamese()
    print(net)
    
    
 #Train

print('initializing training')

loss_function = torch.nn.BCEWithLogitsLoss()#default for size average is true
loss_value = 0
loss_values = []

network = Siamese() #creates a new network
network.to(device)
network.train() #sets the mode of the network to training

optimizer = torch.optim.Adam(network.parameters(), lr = learningRate)
optimizer.zero_grad() #zeros out the gradients

accuracies = []
a_assist = []

start_time = time.time()
initial_start_time = start_time
print('starting training loop')
for i, (imgA, imgB, label) in enumerate(trainLoader, start=1):
    if i > iterations:
        break
        
    
    
    imgA = Variable(imgA.cuda())
    imgB = Variable(imgB.cuda())
    label = Variable(label.cuda())
    
    optimizer.zero_grad() #zeros out the gradients since paramaters already updated with old gradient
    
    output = network.forward(imgA, imgB) #gets similarity probability
    
    loss = loss_function(output, label)
    
    li = loss.item()
    
    loss_value += li
    loss_values.append(li)
    loss.backward() #computes the gradient of loss for all parameters
    
    optimizer.step() #updates parameters
    
    #print updates for the user
    if i % 10 == 0:
        print(f'{i} loss: {loss_value/10} time elapsed: {time.time()-start_time}')
        loss_value = 0
        start_time = time.time()
        if i % 100 == 0:
            correct, wrong = 0, 0
            for _, (testA, testB) in enumerate(testLoader, 1):

                testA, testB = Variable(testA.cuda()), Variable(testB.cuda())
                output = network.forward(testA, testB).data.cpu().numpy() #computes the probability
                prediction = np.argmax(output) #gets the index of highest value in output
                if prediction == 0:
                    correct += 1
                else:
                    wrong += 1

            print('-'*100)
            print(f'{i} Testing Set Correct: {correct} Wrong: {wrong} Precision: {correct*1.0/(correct + wrong)}')
            print('-'*100)
            accuracies.append(correct*1.0/(correct+wrong))
            a_assist.append(i)
            if i % 10000 == 0:
                torch.save(network.state_dict(), f'{modelPath}/training-model-{i}.pt')


print('finish training loop, time elapsed: ', str(time.time()-initial_start_time))
    
#add final accuracies
accuracy = 0.0
counter = 0
for d in accuracies:
    print(d)
    accuracy += d
    counter += 1
print("#"*100)
print("final accuracy: ", accuracy/counter)
    

#define subplots
fig, ax = plt.subplots(2, 1, figsize=(15,20))
#fig.tight_layout()

#create subplots
ax[0].plot(range(1, iterations + 1), loss_values, color='red')
ax[0].set_title('Loss Values During Training')
ax[0].set_ylabel('Loss Value')
ax[0].set_xlabel('Epoch')
ax[1].plot(a_assist, accuracies, color='blue')
ax[1].set_title('Accuracies During Training')
ax[1].set_ylabel('Accuracies')
ax[1].set_xlabel('Epoch')


