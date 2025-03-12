import os
import json
import torch
import numpy as np
import pandas as pd
from torch import optim
from data_utils import create_dataloaders


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

        if torch.isnan(loss).item(): 
            print("NaN loss detected, skipping batch")
            continue
        
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




def validate(model, val_loader, criterion, device):
    """
    Validate model on validation data
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    val_loader : DataLoader
        Validation data loader
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to run validation on
    
    Returns:
    --------
    dict: validation statistics including per-class accuracy
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    class_correct = list(0. for i in range(2))  # Assuming binary classification
    class_total = list(0. for i in range(2))
    
    # For computing metrics
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(images, tabular)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            
    
    # Calculate metrics
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    
    # Calculate additional metrics if sklearn is available
    advanced_metrics = {}
    try:
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
        
        # Convert lists to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        
        # Compute precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Try to compute ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = None
        
        advanced_metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }
        
    except ImportError:
        pass
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'correct': val_correct,
        'total': val_total,
        'advanced_metrics': advanced_metrics,
        'predictions': all_predictions,
        'targets': all_targets
    }




def train(model, train_loader, val_loader, criterion, optimizer, 
                     device, num_epochs, model_dir, checkpoint_freq=1, save_best_only=True, 
                     early_stopping_patience=None, start_epoch=0):
    """
    Train the model with early stopping implemented and checkpointing
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to run training on
    num_epochs : int
        Number of training epochs
    model_dir : str
        Directory to save model results
    checkpoint_freq : int
        Save checkpoint every N epochs (if save_best_only=False)
    save_best_only : bool
        Only save checkpoint if it's the best so far
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping
    start_epoch : int
        Epoch to start training from
    
    Returns:
    --------
    dict: model results
    """
    # Variables for tracking best model and early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    
    # Training history for this model
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_stats = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_stats = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_stats['loss'])
        history['train_acc'].append(train_stats['accuracy'])
        history['val_loss'].append(val_stats['loss'])
        history['val_acc'].append(val_stats['accuracy'])
        
        # Save checkpoints
        is_best = False
        
        # Determine if this is the best model so far
        if val_stats['accuracy'] > best_val_acc or (val_stats['accuracy'] == best_val_acc and val_stats['loss'] < best_val_loss):
            best_val_acc = val_stats['accuracy']
            best_val_loss = val_stats['loss']
            best_epoch = epoch
            is_best = True
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Save checkpoint
        if (not save_best_only and (epoch + 1) % checkpoint_freq == 0) or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_stats['loss'],
                'val_acc': val_stats['accuracy'],
                'train_loss': train_stats['loss'],
                'train_acc': train_stats['accuracy'],
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
            }
            
            
            # Save regular checkpoint
            if not save_best_only and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if is_best:
                best_model_path = os.path.join(model_dir, 'best_model.pth')
                torch.save(checkpoint, best_model_path)
                print(f"Best model saved to {best_model_path}")
        
        # Early stopping
        if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break
    
    # Save model history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Load the best model for final evaluation
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_val_stats = validate(model, val_loader, criterion, device)
    
    # Store results from this model
    model_result = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'final_val_loss': final_val_stats['loss'],
        'final_val_acc': final_val_stats['accuracy'],
        'advanced_metrics': final_val_stats['advanced_metrics']
    }
    
    # Save model results
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(model_result, f, indent=4)
    
    return model_result


