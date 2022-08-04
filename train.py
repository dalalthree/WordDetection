#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print('python: ' + str(sys.version))
print('torch: ' + str(torch.__version__))
print('torchvision: ' + str(torchvision.__version__))


#load Data

def getBatch(batchSize):

#Training Paramaters
iterations = 10000
batchSize = 180
learningRate = 0.00006
gradientDecay = 0.9
gradientDecaySquare = 0.99

#Train

loss_function = torch.nn.BCEWithLogitsLoss()#default for size average is true
loss_value = 0

network = Siamese() #creates a new network
network.to(device)
network.train() #sets the mode of the network to training

optimizer = torch.optim.Adam(network.parameters(), lr = learningRate)
optimizer.zero_grad() #zeros out the gradients


for i in range(iterations):
    pair = getBatch('''insert files here''',batchSize) #gets an image pair
    img1 = pair[0]
    img2 = pair[1]
    label = pair[2]
    
    optimizer.zero_grad() #zeros out the gradients since paramaters already updated with old gradient
    
    output = network.forward(img1, img2) #gets similarity probability
    
    loss = loss_function(output, label)
    
    lose += loss.item()
    loss.backward() #computes the gradient of loss for all parameters
    
    optimizer.step() #updates parameters
    
    #print updates for the user
    if i % 10 == 0:
        print(f'{i} loss: {loss_value/10}')
    if i % 100 = 0:
        torch.save(network.state_dict(), f'{modelPath}/training-model-{i}.pt')
        correct, wrong = 0, 0
        for _, (testA, testB) in enumerate(testingSet, 1):
            testA, testB = Variable(testA.cuda()), Variable(testB.cuda())
            output = network.forward(testA, testB).data.cpu().numpy() #computes the probability
            prediction = np.argmax(output) #gets the index of highest value in output
            if prediction == 0:
                correct += 1
            else:
                wrong += 1
        print('-'*100)
        print(f'{i} Testing Set Correct: {correct} Wrong: {wrong} Precision: {correct*1.0/(correct + error)}')
        print('-'*100)

#add final accuracies
#
#
#
#....................
            
            

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
        #  return self.sigmoid(out)
        return out
    
if __name__ == '__main__':
    net = Siamese()
    print(net)
    
    

#Test Image Loading
Image(filename='./docs/cvl-database-1-1/trainset/pages/0001-1.png') 

#imgT = mpimg.imread('./docs/cvl-database-1-1/trainset/words/0050/0050-8-7-4-Usher.tif', 'r') 
#imgT = mpimg.imread('./docs/cvl-database-1-1/trainset/lines/0001/IMG_3512.jpg', 'r') 
#imgT = mpimg.imread('./docs/cvl-database-1-1/trainset/pages/0001-1.png') 
