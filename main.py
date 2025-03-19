from utils.train import train
from utils.eval import validate
from data.dataloader import create_dataloader
from models.model import CustomNet
import wandb 
import torch

def main(wandb):
    #Get the hyperparameters from wandb
    criterion_wandb = wandb.config.criterion
    optimizer_wandb = wandb.config.optimizer
    num_epochs = wandb.config.epochs
    lr = wandb.config.lr
    momentum = wandb.config.momentum
    batch_size = wandb.config.batch_size
    
    #Setting the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    
    train_loader , val_loader = create_dataloader(batch_size)

    model = CustomNet().to(device)
    
    #initialize the optimizer    
    if(optimizer_wandb == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    #initialize the criterion
    if(criterion_wandb == "CrossEntropy"):
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.NLLLoss()        
        
    best_acc = 0
    #start training and validation loop
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer,device,wandb= wandb)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion,wandb,epoch)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)
        wandb.log({"Best Validation Accuracy":best_acc})

    
    
    
    