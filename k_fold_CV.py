import os
import json
import torch
import numpy as np
import pandas as pd
from torch import optim
from datetime import datetime
from train_utils import train_epoch
from data_utils import create_dataloaders




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
            
            # Per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate metrics
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    # Calculate per-class accuracy
    class_acc = {}
    for i in range(2):
        if class_total[i] > 0:
            class_acc[f'class_{i}'] = class_correct[i] / class_total[i]
        else:
            class_acc[f'class_{i}'] = 0.0
    
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
        'class_accuracy': class_acc,
        'advanced_metrics': advanced_metrics,
        'predictions': all_predictions,
        'targets': all_targets
    }





def train_single_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                     device, num_epochs, fold_dir, checkpoint_freq=1, save_best_only=True, 
                     early_stopping_patience=None, start_epoch=0):
    """
    Train a single fold
    
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
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler (optional)
    device : torch.device
        Device to run training on
    num_epochs : int
        Number of training epochs
    fold_dir : str
        Directory to save fold results
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
    dict: fold results
    """
    # Variables for tracking best model and early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    
    # Training history for this fold
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
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats['loss'])
            else:
                scheduler.step()
        
        # Update history
        history['train_loss'].append(train_stats['loss'])
        history['train_acc'].append(train_stats['accuracy'])
        history['val_loss'].append(val_stats['loss'])
        history['val_acc'].append(val_stats['accuracy'])
        
        print(f"Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.4f}, "
              f"Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.4f}")
        print(f"Class 0 (Benign) Acc: {val_stats['class_accuracy']['class_0']:.4f}, "
              f"Class 1 (Malignant) Acc: {val_stats['class_accuracy']['class_1']:.4f}")
        
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
                'class_acc': val_stats['class_accuracy']
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save regular checkpoint
            if not save_best_only and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(fold_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if is_best:
                best_model_path = os.path.join(fold_dir, 'best_model.pth')
                torch.save(checkpoint, best_model_path)
                print(f"Best model saved to {best_model_path}")
        
        # Early stopping
        if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break
    
    # Save fold history
    with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Load the best model for final evaluation
    best_model_path = os.path.join(fold_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_val_stats = validate(model, val_loader, criterion, device)
    
    # Store results from this fold
    fold_result = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'final_val_loss': final_val_stats['loss'],
        'final_val_acc': final_val_stats['accuracy'],
        'final_class_acc': final_val_stats['class_accuracy'],
        'advanced_metrics': final_val_stats['advanced_metrics']
    }
    
    # Save fold results
    with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
        json.dump(fold_result, f, indent=4)
    
    return fold_result





def train_with_kfold_csv(
    fold_csv_list,
    img_dir,
    img_id_column,
    target_column,
    tabular_columns,
    model,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=10,
    batch_size=32,
    checkpoint_dir='checkpoints',
    checkpoint_freq=1,
    save_best_only=True,
    early_stopping_patience=None,
    resume_from_checkpoint=None
):
    """
    Training with k-fold cross-validation using CSV files to define folds
    
    Parameters:
    -----------
    fold_csv_list : list
        List of CSV filenames, each containing one fold's data
    img_dir : str
        Directory containing the images
    img_id_column : str
        Column name in CSV that contains image IDs/filenames
    target_column : str
        Column name in CSV that contains target labels
    tabular_columns : list
        List of column names to use as tabular features
    model : torch.nn.Module
        PyTorch model that accepts both image and tabular data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler (optional)
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    checkpoint_dir : str
        Directory to save checkpoints
    checkpoint_freq : int
        Save checkpoint every N epochs (if save_best_only=False)
    save_best_only : bool
        Only save checkpoint if it's the best so far
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping
    resume_from_checkpoint : str
        Path to checkpoint to resume training from
    """
    # For reproducibility
    torch.manual_seed(42)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Run ID for this training session
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Store configuration
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'num_folds': len(fold_csv_list),
        'checkpoint_freq': checkpoint_freq,
        'save_best_only': save_best_only,
        'early_stopping_patience': early_stopping_patience,
        'tabular_columns': tabular_columns
    }
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Lists to store metrics for each fold
    fold_results = []
    
    # If CUDA is available, move model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Loop through each fold
    for fold, fold_csv in enumerate(fold_csv_list):
        fold_dir = os.path.join(run_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"FOLD {fold+1}/{len(fold_csv_list)}")
        print(f"Using {fold_csv} for validation")
        print("-----------------------------------")
        
        # Load validation data for this fold
        val_df = pd.read_csv(os.path.join('data', 'k_folds', fold_csv))
        
        # Get training data (all CSVs except current fold)
        train_dfs = []
        for i, csv_file in enumerate(fold_csv_list):
            if i != fold:
                train_dfs.append(pd.read_csv(os.path.join('data', 'k_folds', csv_file)))
        
        if not train_dfs:
            raise ValueError("No training data found. Check fold_csv_list configuration.")
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_df, val_df, img_dir, img_id_column, target_column, 
            tabular_columns, batch_size, num_workers=4
        )
        
        # Reset model for each fold if not the first fold
        if fold > 0:
            # Try to reset model parameters if the model has this method
            if hasattr(model, 'reset_parameters') and callable(model.reset_parameters):
                model.reset_parameters()
            else:
                # Otherwise recreate the model (assuming it can be recreated the same way)
                model_class = type(model)
                if hasattr(model, '__init_params'):
                    model = model_class(**model.__init_params)
                else:
                    print("Warning: Unable to reset model parameters between folds. Using same model instance.")
            
            optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
            if scheduler is not None:
                scheduler = type(scheduler)(optimizer, **scheduler.__dict__)
        
        # Variables to track training progress
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint and fold == 0:  # Only resume for the first fold
            if os.path.isfile(resume_from_checkpoint):
                print(f"Loading checkpoint '{resume_from_checkpoint}'")
                checkpoint = torch.load(resume_from_checkpoint)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Checkpoint loaded. Resuming from epoch {start_epoch+1}")
            else:
                print(f"No checkpoint found at '{resume_from_checkpoint}', starting from scratch")
        
        # Move model to device
        model.to(device)
        
        # Train this fold
        fold_result = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, num_epochs, fold_dir, checkpoint_freq, save_best_only,
            early_stopping_patience, start_epoch
        )
        
        # Add fold number to result
        fold_result['fold'] = fold + 1
        fold_results.append(fold_result)
    
    # Calculate and print average results across all folds
    avg_best_val_acc = np.mean([x['best_val_acc'] for x in fold_results])
    avg_final_val_acc = np.mean([x['final_val_acc'] for x in fold_results])
    
    # Average class accuracies
    avg_class_0_acc = np.mean([x['final_class_acc']['class_0'] for x in fold_results])
    avg_class_1_acc = np.mean([x['final_class_acc']['class_1'] for x in fold_results])
    
    # Try to get average precision, recall, F1
    advanced_metrics = {}
    try:
        avg_precision = np.mean([x['advanced_metrics']['precision'] for x in fold_results])
        avg_recall = np.mean([x['advanced_metrics']['recall'] for x in fold_results])
        avg_f1 = np.mean([x['advanced_metrics']['f1'] for x in fold_results])
        advanced_metrics = {
            'avg_precision': float(avg_precision),
            'avg_recall': float(avg_recall),
            'avg_f1': float(avg_f1)
        }
    except:
        pass
    
    overall_results = {
        'avg_best_val_acc': float(avg_best_val_acc),
        'avg_final_val_acc': float(avg_final_val_acc),
        'avg_class_0_acc': float(avg_class_0_acc),
        'avg_class_1_acc': float(avg_class_1_acc),
        'advanced_metrics': advanced_metrics,
        'fold_results': fold_results
    }
    
    # Save overall results
    with open(os.path.join(run_dir, 'overall_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    print(f"\nCROSS VALIDATION RESULTS FOR {len(fold_csv_list)} FOLDS:")
    print(f"Average Best Validation Accuracy: {avg_best_val_acc:.4f}")
    print(f"Average Final Validation Accuracy: {avg_final_val_acc:.4f}")
    print(f"Average Class 0 (Benign) Accuracy: {avg_class_0_acc:.4f}")
    print(f"Average Class 1 (Malignant) Accuracy: {avg_class_1_acc:.4f}")
    
    if advanced_metrics:
        print(f"Average Precision: {advanced_metrics['avg_precision']:.4f}")
        print(f"Average Recall: {advanced_metrics['avg_recall']:.4f}")
        print(f"Average F1 Score: {advanced_metrics['avg_f1']:.4f}")
    
    return fold_results, run_dir