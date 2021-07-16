#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import torch.nn.functional as F


# In[18]:



logpath = "D:/Datasets/GANS/TensorboardFiles/runs/mnist"
writer = SummaryWriter(logpath)

print("Tensorboard code started...")

device = "cpu"

input_size = 28 * 28 * 1
hidden_size = 128 * 2
num_classes = 10
num_epochs = 2
batch_size = 64
learning_rate = 3e-4


# In[4]:


train_dataset = torchvision.datasets.MNIST(root = "D:/Datasets/GANS",
                                           train = True,
                                           transform = transforms.ToTensor(),
                                           download = True)

test_dataset = torchvision.datasets.MNIST(root = "D:/Datasets/GANS",
                                          train = False,
                                          transform = transforms.ToTensor())


# In[5]:


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)


# In[6]:


examples = iter(test_loader)
examples_data, examples_target = examples.next()


# In[8]:


for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(examples_data[i][0], cmap = 'gray')
#plt.show()
# instead of showing images directly, we add images to tensorboard
img_grid = torchvision.utils.make_grid(examples_data)
writer.add_image('mnist_images', img_grid)
writer.close() # makes sure all the outputs are flushed here
#sys.exit()
# print(image grid is added to tensorboard)


# In[9]:


# fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# In[10]:


model = NeuralNet(input_size, hidden_size, num_classes).to(device)


# In[11]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# adding model graph to tensorboard
writer.add_graph(model, examples_data.reshape(-1, 28*28))
writer.close()
print("Model graph is added to tensorboard...")
#sys.exit()


# In[19]:


n_total_steps = len(train_loader)
running_loss = 0.0 # loss over the training process
running_correct = 0 # correctly predicted images at each iteration in an epoch

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin_shape = (64, 1, 28, 28)
        # resized = (64, 28*28)
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print("Epoch : {}, step : {}, loss : {}".format(epoch+1, i+1, loss.item()))
            # to write a metric in tensorboard
            # writer.add_scaler(label, metric, global_step)
            # global_step = current_epoch * length_of_batches + current_iteration
            writer.add_scalar("training loss", running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar("Running Accuracy", running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
    print()


# In[20]:


# test the model
def results(loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
    
        for images, labels in loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
        
            outputs = model(images)
            # max returns(value, index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
        acc = 100.0 * n_correct / n_samples
        print("Accuracy of the network on the given set of images: {}".format(acc))


# In[21]:


# training results 
results(train_loader)
# testing results
results(test_loader)


# In[ ]:


# to add precision-recall curve in tensorboard
labels = []
preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
    for images, labels1 in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels1 = labels1.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels1.size(0)
        n_correct += (predicted == labels1).sum().item()
        
        class_probs = [F.softmax(output, dim = 0) for output in outputs]
        preds.append(class_probs)
        labels.append(predicted)
    
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)
    
    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step = 0)
        writer.close()
        
print("Added PR curve to the tensorboard...")

