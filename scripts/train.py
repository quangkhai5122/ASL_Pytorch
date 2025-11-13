import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import math
import time
import os
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold # Import KFold
from datetime import datetime

try:
    from scripts.config import (
        DEVICE, N_EPOCHS_PER_FOLD, LR_MAX, N_WARMUP_EPOCHS, OPTIMIZER_LR, OPTIMIZER_WD, CLIP_NORM,
        LABEL_SMOOTHING, SEED, N_FOLDS, VAL_BATCH_SIZE, WD_RATIO
    )
    from scripts.dataset import ASLParquetDataset, AllSignsBatchSampler
    from scripts.model import ASLTransformerModel
except ImportError:
    from config import (
        DEVICE, N_EPOCHS_PER_FOLD, LR_MAX, N_WARMUP_EPOCHS, OPTIMIZER_LR, OPTIMIZER_WD, CLIP_NORM,
        LABEL_SMOOTHING, SEED, N_FOLDS, VAL_BATCH_SIZE, WD_RATIO
    )
    from dataset import ASLParquetDataset, AllSignsBatchSampler
    from model import ASLTransformerModel

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Helper Functions (LR Scheduler, Metrics, Path Resolver)
# =============================================================================
def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS_PER_FOLD):
    WARMUP_METHOD = 'log'
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

def get_lr_scheduler(optimizer):
    # Pre-compute LR schedule for all epochs
    lr_schedule = [lrfn(step, N_WARMUP_EPOCHS, LR_MAX, num_training_steps=N_EPOCHS_PER_FOLD) 
                   for step in range(N_EPOCHS_PER_FOLD)]
    
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
# Cross-Validation Runner 
# =============================================================================
def run_cross_validation(csv_path='data/train.csv', data_root='data/', save_dir='models/', tensorboard_dir='runs/', resume=True):
    print(f"--- Starting ASL Cross-Validation (PyTorch) ---")
    print(f"Device: {DEVICE} | Folds: {N_FOLDS} | Epochs per Fold: {N_EPOCHS_PER_FOLD}")
    
    # 1. Load Dataset
    dataset = ASLParquetDataset(csv_path=csv_path, data_root=data_root)
    if len(dataset) == 0: return

    # 2. Initialize KFold (Deterministic splits)
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_splits = list(kfold.split(dataset.df, dataset.labels))
    
    # 3. Setup Directories và Checkpoint Path
    save_dir_abs = resolve_save_dir(save_dir)
    CV_CHECKPOINT_PATH = os.path.join(save_dir_abs, "cv_checkpoint_latest.pth")
    
    # Khởi tạo trạng thái mặc định
    start_fold = 0
    cv_results = []
    run_name = f"ASL_CV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    # 5. Vòng lặp Cross-Validation
    # Bắt đầu từ start_fold
    for fold_idx in range(start_fold, N_FOLDS):
        print(f"\n{'='*20} FOLD {fold_idx+1}/{N_FOLDS} {'='*20}")
        
        # --- A. Setup DataLoaders ---
        train_indices, val_indices = all_splits[fold_idx]
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_labels_subset = dataset.labels[train_indices]
        train_sampler = AllSignsBatchSampler(train_labels_subset)
        
        train_loader = DataLoader(train_subset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # --- B. Initialize Model, Optimizer, Scheduler ---
        model = ASLTransformerModel().to(DEVICE)
        # Set base_lr=1.0 because scheduler will set LR directly
        optimizer = optim.AdamW(model.parameters(), lr=1.0, weight_decay=OPTIMIZER_WD)
        scheduler = get_lr_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # --- C. Resume Logic (Trong Fold) ---
        start_epoch = 0
        best_val_acc_in_fold = 0.0

        if resume and os.path.exists(CV_CHECKPOINT_PATH):
             # Tải lại checkpoint
             checkpoint = load_checkpoint(CV_CHECKPOINT_PATH, DEVICE, model, optimizer, scheduler)

             if checkpoint and checkpoint.get('current_fold_index') == fold_idx:
                print("[INFO] Resuming training within this fold...")
                start_epoch = checkpoint.get('next_epoch_index', 0)
                best_val_acc_in_fold = checkpoint.get('best_val_acc_in_fold', 0.0)
                print(f"[INFO] Resuming at Epoch {start_epoch + 1}. Best Val Acc so far: {best_val_acc_in_fold:.4f}")

                if start_epoch >= N_EPOCHS_PER_FOLD:
                    print("[INFO] This fold was already completed. Finalizing fold transition.")
                    pass 

        # --- D. Initialize TensorBoard Writer ---
        writer = SummaryWriter(log_dir=os.path.join(tb_base_dir, f'fold_{fold_idx+1}'))

        # --- E. Training Loop cho Fold này ---
        # Bắt đầu từ start_epoch
        for epoch in range(start_epoch, N_EPOCHS_PER_FOLD):
            
            # 0. Update Learning Rate and Adaptive Weight Decay
            scheduler.step(epoch)
            # Update weight decay based on current LR (weight_decay = lr * WD_RATIO)
            new_wd = update_adaptive_weight_decay(optimizer, WD_RATIO)
            
            # 1. Training & Validation
            train_loss, train_acc, train_top5 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch + 1, N_EPOCHS_PER_FOLD)
            val_loss, val_acc, val_top5 = validate_epoch(model, val_loader, criterion, DEVICE, epoch + 1, N_EPOCHS_PER_FOLD)

            # 3. Logging
            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']
            print(f"Epoch {epoch+1}/{N_EPOCHS_PER_FOLD}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}, Weight Decay: {current_wd:.2e}")

            # Ghi vào TensorBoard 
            writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
            writer.add_scalars('Accuracy_Top1', {'Train': train_acc, 'Validation': val_acc}, epoch + 1)
            writer.add_scalars('Accuracy_Top5', {'Train': train_top5, 'Validation': val_top5}, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
            
            # 4. Save Best Model (Fold-specific) - "Best Model"
            if val_acc > best_val_acc_in_fold:
                best_val_acc_in_fold = val_acc
                model_save_path = os.path.join(save_dir_abs, f"model_best_fold_{fold_idx+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"  [SAVE] New best model for Fold {fold_idx+1} saved! Val Acc: {best_val_acc_in_fold:.4f}")

            # 5. Save Latest Checkpoint (Central CV State) - "Last Model"
            checkpoint_state = {
                'run_name': run_name,
                'current_fold_index': fold_idx,
                'next_epoch_index': epoch + 1,
                'fold_results': cv_results,
                'best_val_acc_in_fold': best_val_acc_in_fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            save_checkpoint(CV_CHECKPOINT_PATH, checkpoint_state)
        
        # Kết thúc Fold
        writer.close()
        if len(cv_results) <= fold_idx:
            cv_results.append(best_val_acc_in_fold)
        
        print(f"\nFold {fold_idx+1} Finished. Best Validation Accuracy: {best_val_acc_in_fold:.4f}")
        
        # Cập nhật checkpoint để chuẩn bị cho Fold tiếp theo
        # Đánh dấu fold hiện tại đã hoàn thành và trỏ đến fold tiếp theo.
        checkpoint_state = {
             'run_name': run_name,
             'current_fold_index': fold_idx + 1,
             'next_epoch_index': 0, 
             'fold_results': cv_results,
             # Bỏ qua trạng thái model/optimizer khi chuyển Fold để giữ checkpoint nhẹ khi chuyển giao.
        }
        save_checkpoint(CV_CHECKPOINT_PATH, checkpoint_state)

        # Giải phóng bộ nhớ GPU
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
    run_cross_validation()
