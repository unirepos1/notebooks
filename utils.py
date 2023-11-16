import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from matplotlib import pyplot as plt

import numpy as np



def func0(xy):
    return (xy**2).sum(-1)/10.8822

def func1(xy):
    return ((5 * xy[:,0]**2 + 0.3 * xy[:, 1]**2)/28.8377)

def func2(xy):
    w = -20, 9, -12, 4, -3, -0.7, -0.5, 0.5, 0.3
    x, y = xy[:, 0]*2, xy[:, 1]*3
    return (w[0] * x + w[1] * y + w[2] * x * y + w[3] * x**2 + w[4] * y **2 + 
            w[5] * x**3 + w[6] * y**3 + 0.3 * (x**4 + y**4 - 0.99 * y**2 * x**2) +
           torch.cos(0.3 * x + y * 0.7) * 37 + torch.sin(0.5 * y) * 19)/1175.0842

def func3(xy):
    x,y = xy[:,0], xy[:,1]
    return  (0.5 * (1.5 * x - y) + 0.3 * torch.sin(15 * y * x) + 0.5 * torch.cos(10 * x) + (x**2 + y**2 ))/10.8875

class ML_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = torch.nn.Parameter(torch.tensor(((-1.5, -2),)))


class Batch_loss():
    def __init__(self, loss, bs=0):
        self.bs = bs
        self.loss = loss
        torch.manual_seed(0)
    
    def __call__(self, params):
        loss = self.loss(params)
        if self.bs > 0:
            n_noise = 40
            x = torch.normal(0, 4, (n_noise, 2))
            l = torch.normal(0, np.sqrt(100 / self.bs), (n_noise,))
            ls = torch.normal(0., 1, (n_noise,))

            dists = ((-torch.cdist(params.unsqueeze(0), x.unsqueeze(0))/ls.abs()).exp() * l).sum(-1).squeeze(0)
            loss +=  dists
        return loss

def plot_loss_surface_toy(loss_func):
    l = 4
    x = np.linspace(-l, l, 100)
    y = np.linspace(-l, l, 100)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    
    loss = Batch_loss(loss_func, bs=0)    
    Z = loss(torch.stack([torch.tensor(x.flatten()).float() for x in [X, Y]]).T).reshape(X.shape)
    plt.contourf(X, Y, Z, cmap='viridis', levels=25)
    plt.ylim(-l,l)
    plt.xlim(-l,l)

def train_toy(loss_func,  optim, optim_kwargs, bs=0, nsteps = int(5e2)):  
    model = ML_Model()
    optim_ = optim(model.parameters(), **optim_kwargs)
    param_traj = [list(model.params[0].detach().cpu().numpy())]
    
    loss = Batch_loss(loss_func, bs=bs)
    for i in range(nsteps):
        train_loss = loss(model.params)
        train_loss.backward()
        optim_.step()
        model.zero_grad()
        param_traj.append(list(model.params[0].detach().cpu().numpy()))
        #if ((i + 1) % max(nsteps//10, 1)) == 0:
        #    print(f"Epoch {i + 1} trainloss {train_loss.item():.2e}")
    bs_str = f"batch-size: {bs}" if bs else "full-batch"
    splitter = "'"
    plt.plot(*np.asarray(param_traj).T,  
                        alpha=0.9, label=f'{str(optim).split(splitter)[1]}: {optim_kwargs} {bs_str}')
    
    print(f'{str(optim).split(splitter)[1]}: {optim_kwargs} {bs_str}||loss = {loss_func(model.params).item():.2f}')
    return model.params
    

# Specifiy the LeNet5 architecture
class Lenet5(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, latent_dim=32):
        super().__init__()
        self.feature_extractor =  nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(3,  8, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(8,  16, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16, latent_dim),
        )
        self.lin2 = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.lin2(x)
        return x


def show_samples(dataset: VisionDataset):
    h, w = 5, 10
    fig, ax = plt.subplots(h, w)
    fig.set_size_inches((w, h))
    ax = ax.ravel()
    for i in range(h * w):
        img, label = dataset[i]
        ax[i].imshow(img.permute( (1, 2, 0)), cmap='gray')
        ax[i].axis('off')
    plt.show()


def train_one_epoch(
        model: torch.nn.Sequential,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> [float, float]:

    # put the model into the training mode
    model.train()
    
    losses = []
    predictions = []
    labels = []

    for x, y in train_loader:
        
        # forward pass
        logits = model(x)
        
        loss = criterion(logits, y)

        # do gradient updates
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        
        # collect statistics
        prediction = torch.argmax(logits.detach(), dim=-1)
        predictions.append(prediction)
        labels.append(y)
        
        losses.append(loss.detach())
                
        
    epoch_loss = torch.mean(torch.tensor(losses))

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    accuracy = torch.sum(predictions == labels) / len(predictions)

    return float(epoch_loss), float(accuracy)

@torch.no_grad()
def accuracy(model: torch.nn.Sequential, data_loader: DataLoader) -> [float, float]:

    count = 0
    num_correct = 0

    for x, y in data_loader:
        
        logits = model(x)
        
        yh = torch.argmax(logits, dim=1)
                
        num_correct += (y==yh).float().sum()
        count += x.shape[0]
        
    return float(num_correct / count)