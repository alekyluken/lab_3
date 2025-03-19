from utils.train import train
from utils.eval import validate
from data.dataloader import get_dataloader
from models.model import CustomNet
import wandb 
import torch

def main(wandb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    train_loader , val_loader = get_dataloader(wandb)

    #get the hyperparameters from wandb
    model = CustomNet().to(device)
    criterion = wandb.config.criterion
    optimizer = wandb.config.optimizer
    num_epochs = wandb.config.epochs
    
    
    #start training and validation loop
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer,wandb)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion,wandb,epoch)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)
        wandb.log({"Best Validation Accuracy":best_acc})

    
    
    
    