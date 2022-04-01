import numpy as np

import torch
from torch import nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# shape_in: [M, C_in, H_in, W_in]
# shape_out: [M, C_out, H_out, W_out]
def shape_out_conv2d(shape_in, kernel_size, stride=1, padding=0, dilation=1):
    shape_out = shape_in[-3:]
    shape_out[1:] = [math.floor((s + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1) for s in shape_out[1:]]
    return shape_out

# size: [M, C_in, H_in, W_in]
# size_out: [M, C_out, H_out, W_out]
def shape_out_maxpool2d(shape_in, kernel_size, stride=None, padding=0, dilation=1):
    if stride==None:
        stride = kernel_size
    shape_out = shape_in[-3:]
    shape_out[1:] = [math.floor((s + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1) for s in shape_out[1:]]
    return shape_out

def calc_shape(shape_in, conv2d, maxpool2d):

    shape_out = shape_in[-3:]
    for i in range(len(conv2d)):
        shape_out = shape_out_conv2d(shape_out, conv2d[i][2])
        shape_out = shape_out_maxpool2d(shape_out, maxpool2d[i][0])

    shape_out = shape_out[1] * shape_out[2] * conv2d[-1][1]

    return shape_out

class CNN(nn.Module):
    def __init__(self, shape_in, conv2d_shapes, maxpool2d_shapes, linear_shapes, activation_function, seed=1):
        super(CNN, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        self.conv2d_layers = nn.ModuleList([nn.Conv2d(x[0], x[1], kernel_size=x[2]) for x in conv2d_shapes])
        self.maxpool2d_layers = nn.ModuleList([nn.MaxPool2d(x[0]) for x in maxpool2d_shapes])
        self.linear_layers = nn.ModuleList([nn.Linear(x[0], x[1]) for x in linear_shapes])
        self.activation = activation_function

        self.conv2d_size = len(self.conv2d_layers)
        self.linear_size = len(self.linear_layers)

    def forward(self, x):

        for i in range(self.conv2d_size):
            x = self.conv2d_layers[i](x)
            x = self.maxpool2d_layers[i](x)
            x = self.activation(x)

        x = x.view(-1, self.linear_layers[0].in_features) 

        for i in range(self.linear_size - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)

        x = self.linear_layers[-1](x)
        return x
    
    def train_model(self, loss, lr, batch_size, num_epochs, train_dataset, test_dataset):

        # Loss and optimizer
        self.loss = loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Create PyTorch DataLoader for train and test data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = len(list(test_dataset)), shuffle = False, drop_last = True)
        
        # set our model in the training mode
        self.train()
        
        print("Epoch Loss(train) Loss(test)")
        for epoch in range(num_epochs):

            epoch_loss_train = 0
            # Compute loss for train set
            # Update weights
            for i, batch_sample in enumerate(train_loader):
        
                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
        
                prediction = self(x_batch)

                l = self.loss(prediction, y_batch)
                epoch_loss_train += l.item()

                self.zero_grad()
                l.backward()
                self.optimizer.step()
            epoch_loss_train /= len(train_loader)

            epoch_loss_test = 0
            # Compute loss for test set
            for i, batch_sample in enumerate(test_loader):
        
                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
        
                prediction = self(x_batch)
        
                l = self.loss(prediction, y_batch)
                epoch_loss_test += l.item()
                
            epoch_loss_test /= len(test_loader)
            print(str(epoch) + " " + "{:.6f}".format(epoch_loss_train) + " " + "{:.6f}".format(epoch_loss_test))
