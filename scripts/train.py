import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import math
import time
import os
import argparse
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold # Import KFold
from datetime import datetime
from sklearn.model_selection import train_test_split

try:
    from scripts.config import (
        DEVICE, N_EPOCHS, N_EPOCHS_PER_FOLD, LR_MAX, N_WARMUP_EPOCHS, OPTIMIZER_LR, OPTIMIZER_WD, CLIP_NORM,
        LABEL_SMOOTHING, SEED, N_FOLDS, VAL_BATCH_SIZE, WD_RATIO
    )
    from scripts.dataset import ASLParquetDataset, AllSignsBatchSampler
    from scripts.model import ASLTransformerModel
    from scripts.utils import load_data_maps
except ImportError:
    from config import (
        DEVICE, N_EPOCHS, N_EPOCHS_PER_FOLD, LR_MAX, N_WARMUP_EPOCHS, OPTIMIZER_LR, OPTIMIZER_WD, CLIP_NORM,
        LABEL_SMOOTHING, SEED, N_FOLDS, VAL_BATCH_SIZE, WD_RATIO
    )
    from dataset import ASLParquetDataset, AllSignsBatchSampler
    from model import ASLTransformerModel
    from utils import load_data_maps

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Helper Functions (LR Scheduler, Metrics, Path Resolver)
# =============================================================================
def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
    WARMUP_METHOD = 'log'
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

def get_lr_scheduler(optimizer, total_epochs):
    # Pre-compute LR schedule for all epochs
    lr_schedule = [lrfn(step, N_WARMUP_EPOCHS, LR_MAX, num_training_steps=total_epochs) 
                for step in range(total_epochs)]
    
    # Use LambdaLR with base_lr=1.0 to set LR directly (not as multiplier)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_schedule[epoch] if epoch < len(lr_schedule) else lr_schedule[-1]
    )
    return scheduler

def update_adaptive_weight_decay(optimizer, wd_ratio):
    """
    Update weight decay adaptively based on current learning rate: weight_decay = learning_rate * wd_ratio
    """
    current_lr = optimizer.param_groups[0]['lr']
    new_weight_decay = current_lr * wd_ratio
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = new_weight_decay
    return new_weight_decay

def accuracy(preds, labels):
    return (preds == labels).float().mean()

def top_k_accuracy(output, labels, k=5):
    with torch.no_grad():
        batch_size = labels.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(1.0 / batch_size).item()

def resolve_save_dir(save_dir):
    if os.path.isabs(save_dir):
        save_dir_abs = save_dir
    else:
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir_abs = os.path.join(base_dir, save_dir)
        except NameError:
            save_dir_abs = os.path.abspath(save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)
    return save_dir_abs

# =============================================================================
# Checkpoint Management 
# =============================================================================
def save_checkpoint(path, state):
    """Save training state."""
    torch.save(state, path)

def load_checkpoint(path, device, model=None, optimizer=None, scheduler=None):
    """Load training state and update model/optimizer/scheduler if provided."""
    if not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location=device)
        # Only load model/optimizer state if object is provided and state exists in checkpoint
        if model and checkpoint.get('model_state_dict'):
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint from {path}: {e}")
        return None

# =============================================================================
# Training & Validation Epoch Functions
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num, num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_top5_acc = 0.0
    
    start_time = time.time()

    # Initialize tqdm progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num}/{num_epochs}", unit="batch", leave=False)

    for i, (frames, non_empty_idxs, labels) in progress_bar:
        # Move data to device
        frames = frames.to(device, non_blocking=True)
        non_empty_idxs = non_empty_idxs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(frames, non_empty_idxs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient Clipping
        if CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_acc += accuracy(preds, labels).item()
        running_top5_acc += top_k_accuracy(outputs, labels, k=5)

        # Cập nhật TQDM postfix
        avg_loss = running_loss / (i + 1)
        avg_acc = running_acc / (i + 1)
        progress_bar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{avg_acc:.4f}')

    # Close the progress bar
    progress_bar.close()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_top5_acc = running_top5_acc / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_top5_acc

def validate_epoch(model, dataloader, criterion, device, epoch_num, num_epochs):
    model.eval() 
    running_loss = 0.0
    running_acc = 0.0
    running_top5_acc = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Val Ep {epoch_num}/{num_epochs}", unit="batch", leave=False)

    with torch.no_grad(): 
        for i, (frames, non_empty_idxs, labels) in progress_bar:
            frames = frames.to(device, non_blocking=True)
            non_empty_idxs = non_empty_idxs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames, non_empty_idxs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += accuracy(preds, labels).item()
            running_top5_acc += top_k_accuracy(outputs, labels, k=5)

            avg_loss = running_loss / (i + 1)
            avg_acc = running_acc / (i + 1)
            progress_bar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{avg_acc:.4f}')

    progress_bar.close()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_top5_acc = running_top5_acc / len(dataloader)

    return epoch_loss, epoch_acc, epoch_top5_acc

# =============================================================================
# Train/Val Runner (train on full train_landmark, validate on test_landmark)
# =============================================================================
def run_full_training(train_csv='data/train.csv', val_csv='data/test.csv', data_root='data/', save_dir='models/', tensorboard_dir='runs/', resume=True):
    print(f"Starting ASL Full Training (PyTorch)")
    print(f"Device: {DEVICE} | Epochs: {N_EPOCHS}")
    
    # 1. Load and Merge Datasets
    print("Loading and merging datasets...")
    try:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(val_csv)

        if 'sign' not in df_train.columns or 'sign' not in df_test.columns:
            raise ValueError("CSV files must contain 'sign' column.")

        # Merge
        df_full = pd.concat([df_train, df_test], ignore_index=True)
        print(f"Merged Train ({len(df_train)}) + Test ({len(df_test)}) = Total ({len(df_full)})")
        
        # Create Sign Map from full data
        signs = sorted(df_full['sign'].unique())
        sign2ord_map = {sign: i for i, sign in enumerate(signs)}
        
        # Stratified Split (85% Train, 15% Val)
        train_df, val_df = train_test_split(df_full, test_size=0.15, stratify=df_full['sign'], random_state=42)
        print(f"Split: Train ({len(train_df)}) | Val ({len(val_df)})")
        
    except Exception as e:
        print(f"Error loading/merging data: {e}")
        return

    # 2. Create Datasets
    train_dataset = ASLParquetDataset(data_source=train_df, data_root=data_root, sign2ord_map=sign2ord_map, augment=True)
    val_dataset = ASLParquetDataset(data_source=val_df, data_root=data_root, sign2ord_map=sign2ord_map, augment=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Training or validation dataset is empty.")
        return

    # 3. DataLoader setup
    num_workers = min(os.cpu_count() if os.cpu_count() else 0, 8)
    pin_memory = True if DEVICE.type == 'cuda' else False
    
    train_sampler = AllSignsBatchSampler(train_dataset.labels)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # 4. Setup Directories & checkpoint paths
    save_dir_abs = resolve_save_dir(save_dir)
    checkpoint_path = os.path.join(save_dir_abs, "full_training_checkpoint_latest.pth")
    best_model_path = os.path.join(save_dir_abs, "model_best_full_training.pth")

    run_name = f"ASL_FULL_MERGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_base_dir = os.path.join(tensorboard_dir, run_name)
    
    # 5. Initialize Model, Optimizer, Scheduler
    model = ASLTransformerModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=OPTIMIZER_WD) # Reduced initial LR from 1.0 to 1e-3
    
    # Switch to ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # 6. Resume Logic
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, DEVICE, model, optimizer, scheduler)
        if checkpoint:
            start_epoch = checkpoint.get('next_epoch_index', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            run_name = checkpoint.get('run_name', run_name)
            tb_base_dir = os.path.join(tensorboard_dir, run_name)
            print(f"[INFO] Resuming training at Epoch {start_epoch + 1}. Best Val Acc: {best_val_acc:.4f}")
        else:
            print("[INFO] Checkpoint found but failed to load. Starting from scratch.")

    os.makedirs(tb_base_dir, exist_ok=True)
    print(f"\nTensorBoard logs: {os.path.join(tb_base_dir, 'full_run')}\nRun: tensorboard --logdir={tensorboard_dir}\n")
    writer = SummaryWriter(log_dir=os.path.join(tb_base_dir, 'full_run'))

    # Prime scheduler/weight decay so LR matches schedule before entering the loop
    update_adaptive_weight_decay(optimizer, WD_RATIO)

    # 7. Training Loop
    for epoch in range(start_epoch, N_EPOCHS):
        
        # Training & Validation
        train_loss, train_acc, train_top5 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch + 1, N_EPOCHS)
        val_loss, val_acc, val_top5 = validate_epoch(model, val_loader, criterion, DEVICE, epoch + 1, N_EPOCHS)

        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0]['weight_decay']
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}, Weight Decay: {current_wd:.2e}")

        # TensorBoard logging
        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
        writer.add_scalars('Accuracy_Top1', {'Train': train_acc, 'Validation': val_acc}, epoch + 1)
        writer.add_scalars('Accuracy_Top5', {'Train': train_top5, 'Validation': val_top5}, epoch + 1)
        writer.add_scalar('LearningRate', current_lr, epoch + 1)

        # Update best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  [SAVE] New best model saved! Val Acc: {best_val_acc:.4f}")

        # Step LR/WD for next epoch
        scheduler.step(val_acc)
        update_adaptive_weight_decay(optimizer, WD_RATIO)

        # Save latest checkpoint
        checkpoint_state = {
            'run_name': run_name,
            'next_epoch_index': epoch + 1,
            'best_val_acc': best_val_acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        save_checkpoint(checkpoint_path, checkpoint_state)

    writer.close()
    print(f"\nTraining Finished. Best Validation Accuracy: {best_val_acc:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# Cross-Validation Runner 
# =============================================================================
def run_cross_validation(train_csv='data/train.csv', val_csv='data/test.csv', data_root='data/', save_dir='models_cv/', tensorboard_dir='runs/', resume=True, folds_to_run=None):
    print(f"--- Starting ASL Cross-Validation (PyTorch) ---")
    print(f"Device: {DEVICE} | Folds: {N_FOLDS} | Epochs per Fold: {N_EPOCHS_PER_FOLD}")
    if folds_to_run:
        print(f"Running specific folds: {folds_to_run}")

    # 1. Load and Merge Datasets
    print("Loading and merging datasets...")
    try:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(val_csv)
        if 'sign' not in df_train.columns or 'sign' not in df_test.columns:
            raise ValueError("CSV files must contain 'sign' column.")
        df_full = pd.concat([df_train, df_test], ignore_index=True)
        print(f"Merged Train ({len(df_train)}) + Test ({len(df_test)}) = Total ({len(df_full)})")
        signs = sorted(df_full['sign'].unique())
        sign2ord_map = {sign: i for i, sign in enumerate(signs)}
        
    except Exception as e:
        print(f"Error loading/merging data: {e}")
        return

    # 2. Create Dataset Instances
    # We need two instances: one for training (with augmentation) and one for validation (without)
    # Both use the FULL merged dataframe. Splitting happens via KFold indices.
    dataset_train = ASLParquetDataset(data_source=df_full, data_root=data_root, sign2ord_map=sign2ord_map, augment=True)
    dataset_val = ASLParquetDataset(data_source=df_full, data_root=data_root, sign2ord_map=sign2ord_map, augment=False)
    
    if len(dataset_train) == 0: return

    # 3. Initialize KFold (Deterministic splits)
    # Use dataset_train for splitting (indices are same for both)
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_splits = list(kfold.split(dataset_train.df, dataset_train.labels))
    
    # 4. Setup Directories & Checkpoint Path
    save_dir_abs = resolve_save_dir(save_dir)
    CV_CHECKPOINT_PATH = os.path.join(save_dir_abs, "cv_checkpoint_latest.pth")
    
    # Initialize default state
    cv_results = []
    run_name = f"ASL_CV_MERGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 4. Resume Logic (Global CV State)
    is_resuming = False 
    if resume and os.path.exists(CV_CHECKPOINT_PATH):
        print(f"\n[INFO] Found CV checkpoint. Attempting to resume.")
        # Tải trạng thái CV (chưa cần tải model/optimizer ở đây)
        checkpoint = load_checkpoint(CV_CHECKPOINT_PATH, DEVICE)
        if checkpoint:
            start_fold = checkpoint.get('current_fold_index', 0)
            cv_results = checkpoint.get('fold_results', [])
            run_name = checkpoint.get('run_name', run_name) 
            is_resuming = True
            print(f"[INFO] Resuming Cross-Validation at Fold {start_fold + 1}. Completed folds: {len(cv_results)}")
            
            if start_fold >= N_FOLDS:
                print("[INFO] Cross-Validation already completed according to checkpoint.")
                start_fold = N_FOLDS 
        else:
            print("[INFO] Checkpoint file corrupted or loading failed. Starting from scratch.")
            # Reset trạng thái nếu tải thất bại
            start_fold = 0; cv_results = []; run_name = f"ASL_CV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not is_resuming:
        print("\n[INFO] Starting Cross-Validation from scratch.")

    # Setup TensorBoard Directory 
    tb_base_dir = os.path.join(tensorboard_dir, run_name)
    os.makedirs(tb_base_dir, exist_ok=True)
    print(f"\nTensorBoard logs: {tb_base_dir}\nRun: tensorboard --logdir={tensorboard_dir}\n")

    # Cấu hình DataLoader chung
    num_workers = min(os.cpu_count() if os.cpu_count() else 0, 8)
    pin_memory = True if DEVICE.type == 'cuda' else False
    
    # Determine which folds to process
    if folds_to_run is not None:
        folds_indices = [f - 1 for f in folds_to_run] # User passes 1-based, convert to 0-based
    else:
        folds_indices = range(N_FOLDS)

    # 5. Cross-Validation Loop
    for fold_idx in folds_indices:
        if fold_idx < 0 or fold_idx >= N_FOLDS:
            print(f"Skipping invalid fold index: {fold_idx + 1}")
            continue

        print(f"\n{'='*20} FOLD {fold_idx+1}/{N_FOLDS} {'='*20}")
        
        # --- A. Setup DataLoaders ---
        train_indices, val_indices = all_splits[fold_idx]
        train_subset = Subset(dataset_train, train_indices)
        val_subset = Subset(dataset_val, val_indices)
        
        train_labels_subset = dataset_train.labels[train_indices]
        train_sampler = AllSignsBatchSampler(train_labels_subset)
        
        # Config common params to match run_full_training
        num_workers = min(os.cpu_count() if os.cpu_count() else 0, 8)
        pin_memory = True if DEVICE.type == 'cuda' else False

        train_loader = DataLoader(train_subset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # --- B. Initialize Model, Optimizer, Scheduler ---
        model = ASLTransformerModel().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=OPTIMIZER_WD)
        
        # Use ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # --- C. Resume Logic (Fold-Specific) ---
        start_epoch = 0
        best_val_acc_in_fold = 0.0
        
        FOLD_CHECKPOINT_PATH = os.path.join(save_dir_abs, f"checkpoint_fold_{fold_idx+1}.pth")
        
        if resume and os.path.exists(FOLD_CHECKPOINT_PATH):
            checkpoint = load_checkpoint(FOLD_CHECKPOINT_PATH, DEVICE, model, optimizer, scheduler)
            if checkpoint:
                start_epoch = checkpoint.get('next_epoch_index', 0)
                best_val_acc_in_fold = checkpoint.get('best_val_acc_in_fold', 0.0)
                print(f"[INFO] Resuming FOLD {fold_idx+1} at Epoch {start_epoch + 1}. Best Val Acc: {best_val_acc_in_fold:.4f}")
        
        # Prime scheduler/weight decay
        # scheduler.step(0) removed for ReduceLROnPlateau
        update_adaptive_weight_decay(optimizer, WD_RATIO)

        # --- D. Initialize TensorBoard Writer ---
        # Include fold in run name to separate logs
        fold_run_name = f"{run_name}_fold{fold_idx+1}"
        tb_fold_dir = os.path.join(tensorboard_dir, fold_run_name)
        os.makedirs(tb_fold_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_fold_dir)

        # --- E. Training Loop ---
        for epoch in range(start_epoch, N_EPOCHS_PER_FOLD):
            
            # 1. Training & Validation
            train_loss, train_acc, train_top5 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch + 1, N_EPOCHS_PER_FOLD)
            val_loss, val_acc, val_top5 = validate_epoch(model, val_loader, criterion, DEVICE, epoch + 1, N_EPOCHS_PER_FOLD)

            # 3. Logging
            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']
            print(f"Epoch {epoch+1}/{N_EPOCHS_PER_FOLD}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}, Weight Decay: {current_wd:.2e}")

            # TensorBoard
            writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
            writer.add_scalars('Accuracy_Top1', {'Train': train_acc, 'Validation': val_acc}, epoch + 1)
            writer.add_scalars('Accuracy_Top5', {'Train': train_top5, 'Validation': val_top5}, epoch + 1)
            writer.add_scalar('LearningRate', current_lr, epoch + 1)
            
            # 4. Save Best Model
            if val_acc > best_val_acc_in_fold:
                best_val_acc_in_fold = val_acc
                model_save_path = os.path.join(save_dir_abs, f"model_best_fold_{fold_idx+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"  [SAVE] New best model for Fold {fold_idx+1} saved! Val Acc: {best_val_acc_in_fold:.4f}")

            # Step Scheduler & WD
            scheduler.step(val_acc)
            update_adaptive_weight_decay(optimizer, WD_RATIO)

            # 5. Save Fold Checkpoint
            checkpoint_state = {
                'run_name': run_name,
                'fold_index': fold_idx,
                'next_epoch_index': epoch + 1,
                'best_val_acc_in_fold': best_val_acc_in_fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            save_checkpoint(FOLD_CHECKPOINT_PATH, checkpoint_state)
        
        writer.close()
        print(f"\nFold {fold_idx+1} Finished. Best Validation Accuracy: {best_val_acc_in_fold:.4f}")
        
        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. Tổng kết CV
    print(f"\n{'='*20} Cross-Validation Finished {'='*20}")
    
    print("Best Validation Accuracy per Fold:")
    for i, acc in enumerate(cv_results):
        print(f"  Fold {i+1}: {acc:.4f}")
    
    if cv_results:
        mean_acc = np.mean(cv_results)
        std_acc = np.std(cv_results)
        print(f"\nMean CV Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASL Training Script")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "cv"], help="Training mode: 'full' for full data, 'cv' for cross-validation")
    parser.add_argument("--folds", type=int, nargs="+", help="Specific folds to run (1-based), e.g., --folds 1 2 3")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full_training()
    elif args.mode == "cv":
        run_cross_validation(folds_to_run=args.folds)
