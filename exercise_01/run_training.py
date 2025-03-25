
import sys
sys.path.append("../")

import torch.nn as nn

import torch
import numpy as np

from load_data import get_data, get_batch_loaders
from models import CNNModel
from util import run_training_loop, get_device
from tqdm import tqdm



num_epochs = 10

def main():
    spectra, labels = get_data() # spectra shape (8914, 16384)
    
    device = get_device()
    
    spectra = torch.tensor(spectra.astype(np.float32)).to(device)
    labels = torch.tensor(labels.astype(np.float32)).to(device)

    
    model = CNNModel(n_labels=3)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.functional.mse_loss
    
    train_losses = []
    val_losses = []
    
    train_batch_loader, val_batch_loader, test_batch_loader = get_batch_loaders(spectra, labels)
    
    for epoch in tqdm(range(num_epochs)):
        
        train_loss = 0
        # currently no batches are run, needs dataloader
        for batch_idx, (batch_x, batch_y) in enumerate(train_batch_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            forward_pass_outputs = model(batch_x)
            loss = criterion(forward_pass_outputs.ravel(), batch_y.ravel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_losses.append(train_loss/len(train_batch_loader))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_batch_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_batch_loader)
        
        if epoch > 0:
            if val_loss < val_losses[-1]:
                torch.save(model.state_dict(), "saved_models/best_model.pth")

        val_losses.append(val_loss)
            
    print(train_losses)

if __name__ == "__main__":
    main()