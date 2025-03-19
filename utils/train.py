import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from typing import Union, Any
def train(epoch: int,model: Module,train_loader: DataLoader,criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.NLLLoss],optimizer: Optimizer,device: torch.device,wandb: any) -> None:    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 1000 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%',"epoch:",epoch)
    wandb.log({"Train Accuracy":train_accuracy,"Train Loss":train_loss})