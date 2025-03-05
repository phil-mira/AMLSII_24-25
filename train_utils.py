import torch


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to run training on
    
    Returns:
    --------
    dict: epoch statistics
    """
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        tabular = batch['tabular'].to(device)
        targets = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images, tabular)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record statistics
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().item()
        
        # Print batch progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Calculate epoch statistics
    epoch_loss = train_loss / train_total
    epoch_acc = train_correct / train_total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'correct': train_correct,
        'total': train_total
    }



