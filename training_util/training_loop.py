import torch
import torch.nn.functional as F

from tqdm import tqdm


from util.utils import get_device

device = get_device()

def run_training_loop(model, train_loader, num_epochs=3, lr=0.001,):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(iter(train_loader)):
            batch = batch.to(device)
            forward_pass_outputs = model(batch)
            loss = F.mse_loss(forward_pass_outputs.ravel(), batch.y.ravel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()