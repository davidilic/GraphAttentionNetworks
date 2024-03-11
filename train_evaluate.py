import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def train_model(model: nn.Module, data, optimizer, epochs: int, patience: int = 10):
    best_loss = float('inf')
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model((data.x, data.edge_index))
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve == patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break

def evaluate_model(model: nn.Module, data) -> float:
    model.eval()
    with torch.no_grad():
        output = model((data.x, data.edge_index))
        pred = output.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        total = data.test_mask.sum().item()
    return correct / total