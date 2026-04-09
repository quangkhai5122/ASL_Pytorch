import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from scripts.config import *
except ImportError:
    from config import *

# =============================================================================
# Initializers
# =============================================================================
def init_glorot_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_he_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') 
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# =============================================================================
# Landmark Embedding
# =============================================================================

class LandmarkEmbedding(nn.Module):
    def __init__(self, input_dim, units, name):
        super(LandmarkEmbedding, self).__init__()
        self.units = units
        
        # Embedding for missing landmark
        self.empty_embedding = nn.Parameter(torch.zeros(self.units), requires_grad=True)
        
        # Dense layers 
        self.dense = nn.Sequential(
            nn.Linear(input_dim, self.units, bias=False), # dense_1
            nn.GELU(),
            nn.Linear(self.units, self.units, bias=False), # dense_2
        )
        
        # Apply initializations
        self.dense[0].apply(init_glorot_uniform)
        self.dense[2].apply(init_he_uniform)

    def forward(self, x):
        # Check if landmark is missing (sum across feature dimension is 0)
        is_missing = torch.sum(x, dim=2, keepdim=True) == 0
        
        embedded_x = self.dense(x)
        
        # If missing, use the empty_embedding
        # Expand empty_embedding for broadcasting
        empty_emb_expanded = self.empty_embedding.expand_as(embedded_x)
        return torch.where(is_missing, empty_emb_expanded, embedded_x)

# =============================================================================
# Main Embedding Layer
# =============================================================================

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        
        # Positional Embedding 
        self.positional_embedding = nn.Embedding(INPUT_SIZE + 1, UNITS)
        nn.init.zeros_(self.positional_embedding.weight)
        
        # Landmark Embeddings Definitions (Input Dims: Lips: 80, Hand: 42, Pose: 10)
        self.lips_embedding = LandmarkEmbedding(40*2, LIPS_UNITS, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(21*2, HANDS_UNITS, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(5*2, POSE_UNITS, 'pose')
        
        # Landmark Weights (Initialized with zeros)
        self.landmark_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)
        
        # Fully Connected Layers
        assert LIPS_UNITS == HANDS_UNITS == POSE_UNITS, "All landmark units must be the same size"
        
        self.fc = nn.Sequential(
            nn.Linear(LIPS_UNITS, UNITS, bias=False), # fully_connected_1
            nn.GELU(),
            nn.Linear(UNITS, UNITS, bias=False), # fully_connected_2
        )
        
        # Apply initializations
        self.fc[0].apply(init_glorot_uniform)
        self.fc[2].apply(init_he_uniform)

    def forward(self, lips0, left_hand0, pose0, non_empty_frame_idxs):
        # Embeddings
        lips_embedding = self.lips_embedding(lips0)
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        pose_embedding = self.pose_embedding(pose0)
        
        # Merge Embeddings using weighted sum
        x = torch.stack((lips_embedding, left_hand_embedding, pose_embedding), dim=3)
        
        # Apply softmax to weights
        weights = F.softmax(self.landmark_weights, dim=0)
        
        # Weighted sum
        x = torch.sum(x * weights, dim=3)
        
        # Fully Connected Layers
        x = self.fc(x)
        
        # Add Positional Embedding 
        
        # Find max frame index, clipped at 1
        max_frame_idxs = torch.clamp(
            torch.max(non_empty_frame_idxs, dim=1, keepdim=True)[0],
            min=1.0
        )
        
        # Normalize indices
        normalized_idxs = ((non_empty_frame_idxs / max_frame_idxs) * INPUT_SIZE).long()
        
        # Where original index was -1 (padding), use INPUT_SIZE (the padding index)
        positional_indices = torch.where(
            non_empty_frame_idxs == -1.0,
            INPUT_SIZE,
            normalized_idxs
        )
        
        x = x + self.positional_embedding(positional_indices)
        
        return x

# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        # Lists of Dense layers (as in the original implementation)
        self.wq = nn.ModuleList([nn.Linear(d_model, self.depth) for _ in range(num_heads)])
        self.wk = nn.ModuleList([nn.Linear(d_model, self.depth) for _ in range(num_heads)])
        self.wv = nn.ModuleList([nn.Linear(d_model, self.depth) for _ in range(num_heads)])
        
        self.wo = nn.Linear(self.depth * num_heads, d_model)
        
        # Initialization 
        self.apply(init_glorot_uniform)

    def scaled_dot_product(self, q, k, v, attention_mask):
        # attention_mask shape: (Batch, SeqLen, 1) - 1 for valid, 0 for masked
        
        qkt = torch.matmul(q, k.transpose(-2, -1)) # (Batch, SeqLen, SeqLen)
        dk = q.shape[-1]
        scaled_qkt = qkt / math.sqrt(dk)
        
        if attention_mask is not None:
            # Replicating behavior: Mask out frames that should be ignored (Keys).
            # Mask (B, T, 1) -> Expand to (B, 1, T) to mask the Key dimension (columns)
            mask = attention_mask.transpose(-2, -1) # (B, 1, T)
            
            # Apply mask: Where mask is 0, add a large negative number
            scaled_qkt = scaled_qkt + (1.0 - mask) * -1e9

        softmax_weights = F.softmax(scaled_qkt, dim=-1)
        z = torch.matmul(softmax_weights, v)
        return z

    def forward(self, x, attention_mask):
        
        multi_attn = []
        for i in range(self.num_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            
            attn_output = self.scaled_dot_product(Q, K, V, attention_mask)
            multi_attn.append(attn_output)
            
        # Concatenate heads
        multi_head = torch.cat(multi_attn, dim=-1)
        
        # Final linear layer
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention

class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block. 
    """
    def __init__(self, d_model, num_heads, mlp_ratio, dropout_rate):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Multi Layer Perceptron
        mlp_hidden_units = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_units, d_model),
        )
        
        # Apply initializations
        self.mlp[0].apply(init_glorot_uniform)
        self.mlp[3].apply(init_he_uniform)

    def forward(self, x, attention_mask):
        # Residual connections only
        # x = x + mha(x, attention_mask)
        # x = x + mlp(x)
        
        attn_output = self.mha(x, attention_mask)
        x = x + attn_output
        
        mlp_output = self.mlp(x)
        x = x + mlp_output
        
        return x

class Transformer(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, mlp_ratio, dropout_rate):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout_rate)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, attention_mask):
        for block in self.blocks:
            x = block(x, attention_mask)
        return x

# =============================================================================
# Final Model Definition
# =============================================================================

class ASLTransformerModel(nn.Module):
    def __init__(self):
        super(ASLTransformerModel, self).__init__()
        
        self.embedding = Embedding()
        self.transformer = Transformer(
            num_blocks=NUM_BLOCKS,
            d_model=UNITS,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
            dropout_rate=MLP_DROPOUT_RATIO
        )
        
        self.dropout = nn.Dropout(CLASSIFIER_DROPOUT_RATIO)
        self.classifier = nn.Linear(UNITS, NUM_CLASSES)
        
        # Initialize classifier (Glorot Uniform)
        self.classifier.apply(init_glorot_uniform)
            
        # Register normalization constants as buffers
        # Helper to ensure STD is safe (replace 0 with 1 for safe division, as STD=0 in config)
        def make_std_safe(std_np):
            std = torch.tensor(std_np, dtype=torch.float32)
            return torch.where(std == 0, 1.0, std)

        self.register_buffer('LIPS_MEAN', torch.tensor(LIPS_MEAN, dtype=torch.float32))
        self.register_buffer('LIPS_STD', make_std_safe(LIPS_STD))
        self.register_buffer('LEFT_HANDS_MEAN', torch.tensor(LEFT_HANDS_MEAN, dtype=torch.float32))
        self.register_buffer('LEFT_HANDS_STD', make_std_safe(LEFT_HANDS_STD))
        self.register_buffer('POSE_MEAN', torch.tensor(POSE_MEAN, dtype=torch.float32))
        self.register_buffer('POSE_STD', make_std_safe(POSE_STD))

    def normalize(self, tensor, mean, std):
        # Apply normalization using the safe STD buffers.
        normalized = (tensor - mean) / std
        return torch.where(tensor == 0.0, 0.0, normalized)

    def forward(self, frames, non_empty_frame_idxs):
        # 1. Masking
        mask0 = (non_empty_frame_idxs != -1).float().unsqueeze(2) # (B, T, 1)
        
        # Random Frame Masking (Augmentation) - only during training
        if self.training:
            random_noise = torch.rand(mask0.shape, device=mask0.device)
            condition = (random_noise > FRAME_MASK_RATIO) & (mask0 != 0.0)
            mask = torch.where(condition, 1.0, 0.0)
            
            # Correct samples which are now completely masked
            mask_sum = torch.sum(mask, dim=[1, 2], keepdim=True)
            mask = torch.where(mask_sum == 0.0, mask0, mask)
        else:
            mask = mask0

        # 2. Feature Extraction and Normalization
        # Slice to keep only X and Y coordinates
        x = frames[:, :, :, :2]
        
        # Define slicing indices
        LIPS_END = LIPS_START + 40
        LEFT_HAND_END = LEFT_HAND_START + 21
        POSE_END = POSE_START + 5

        # LIPS
        lips = x[:, :, LIPS_START:LIPS_END, :]
        lips = self.normalize(lips, self.LIPS_MEAN, self.LIPS_STD)
        
        # LEFT HAND
        left_hand = x[:, :, LEFT_HAND_START:LEFT_HAND_END, :]
        left_hand = self.normalize(left_hand, self.LEFT_HANDS_MEAN, self.LEFT_HANDS_STD)
        
        # POSE
        pose = x[:, :, POSE_START:POSE_END, :]
        pose = self.normalize(pose, self.POSE_MEAN, self.POSE_STD)
        
        # 3. Flatten spatial dimensions
        B = x.shape[0]
        lips = lips.reshape(B, INPUT_SIZE, -1)
        left_hand = left_hand.reshape(B, INPUT_SIZE, -1)
        pose = pose.reshape(B, INPUT_SIZE, -1)
        
        # 4. Embedding
        x = self.embedding(lips, left_hand, pose, non_empty_frame_idxs)
        
        # 5. Transformer Encoder
        x = self.transformer(x, mask)
        
        # 6. Pooling (Masked average pooling)
        x_sum = torch.sum(x * mask, dim=1)
        mask_sum = torch.sum(mask, dim=1)
        
        # Ensure no division by zero
        mask_sum = torch.clamp(mask_sum, min=1e-9)
        
        x = x_sum / mask_sum
        
        # 7. Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        # Return logits (PyTorch CrossEntropyLoss expects logits)
        return x