"""
DINOv3-based encoder for CT scan processing.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms
from transformers.models.dinov3_vit import DINOv3ViTModel


class DINOv3_Encoder(nn.Module):
    """
    DINOv3 Vision Transformer Encoder for CT scan processing
    
    This wrapper adapts DINOv3 for multi-slice CT processing by:
    1. Processing each RGB image (3 slices) through DINOv3
    2. Aggregating features across all images 
    3. Outputting features compatible with the hypernet framework
    """
    
    def __init__(
        self,
        model_path,
        feature_dim=768,
        num_images=55,  # 165 slices / 3 = 55 RGB images
        aggregation='mean',  # 'mean', 'max', 'cls_token'
        freeze_base=True,
        train_mlp_head=False,
    ):
        super().__init__()

        self.model_path = model_path
        self.feature_dim = feature_dim
        self.num_images = num_images
        self.aggregation = aggregation
        self.train_mlp_head = train_mlp_head
        self.supports_volume_mask = True
        
        # Load DINOv3 model
        try:
            self.dinov3 = DINOv3ViTModel.from_pretrained(model_path)
            print(f"✅ DINOv3 model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load DINOv3 model from {model_path}: {e}")
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            print("❄️ DINOv3 base model frozen")

        # ImageNet normalization only (no resizing; inputs remain at 144x144)
        self.preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Task-agnostic classification head, adapted via LoRA-modified features
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1)
        )

        if train_mlp_head:
            for param in self.mlp_head.parameters():
                param.requires_grad = True
            print("🟢 DINOv3 mlp_head left trainable for joint optimization")
        elif freeze_base:
            for param in self.mlp_head.parameters():
                param.requires_grad = False
            print("❄️ DINOv3 mlp_head frozen (LoRA-only adaptation)")

        print(f"🏗️ DINOv3_Encoder initialized:")
        print(f"   • Feature dimension: {feature_dim}")
        print(f"   • Number of RGB images: {num_images}")
        print(f"   • Aggregation method: {aggregation}")
        print(f"   • Input size: 144x144")
    
    def forward(self, x, mask=None):
        """
        Forward pass through DINOv3 encoder
        
        Args:
            x: Input tensor of shape (batch_size, 3, num_images, height, width)
               where each triplet of consecutive slices forms an RGB image
               
        Returns:
            logits: Classification scores of shape (batch_size, 1)
        """
        aggregated_features = self._compute_aggregated_features(x, mask=mask)
        logits = self.mlp_head(aggregated_features)
        return logits

    def _compute_aggregated_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Shared helper for computing aggregated DINOv3 features."""
        batch_size = x.shape[0]
        num_images = x.shape[2]

        if num_images != self.num_images:
            # Update cached num_images to match dynamic input length
            self.num_images = num_images

        if mask is not None:
            if mask.dim() != 2 or mask.shape[0] != batch_size:
                raise ValueError(f"Mask shape {mask.shape} incompatible with batch size {batch_size}")
            if mask.shape[1] != num_images:
                raise ValueError(f"Mask length {mask.shape[1]} incompatible with num_images {num_images}")
            mask = mask.to(x.device)
            mask_bool = mask.to(dtype=torch.bool)
            mask_float = mask_bool.float()
        else:
            mask_bool = None
            mask_float = torch.ones(batch_size, num_images, device=x.device)

        # Reshape to process each RGB image separately
        x_reshaped = x.contiguous().view(batch_size * num_images, 3, x.shape[-2], x.shape[-1])

        # Apply preprocessing transforms
        x_preprocessed = self.preprocess(x_reshaped)

        outputs = self.dinov3(pixel_values=x_preprocessed)

        # Get last hidden states
        last_hidden_states = outputs.last_hidden_state

        # Extract CLS token features (first token)
        cls_features = last_hidden_states[:, 0, :]

        # Reshape back to separate batch and image dimensions
        cls_features = cls_features.view(batch_size, num_images, self.feature_dim)

        # Aggregate features across images
        if self.aggregation == 'mean':
            counts = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            aggregated_features = (cls_features * mask_float.unsqueeze(-1)).sum(dim=1) / counts
            zero_counts = (mask_float.sum(dim=1) == 0)
            if zero_counts.any():
                aggregated_features[zero_counts] = cls_features[zero_counts, 0, :]
        elif self.aggregation == 'max':
            if mask_bool is not None:
                masked_cls = cls_features.masked_fill(~mask_bool.unsqueeze(-1), float('-inf'))
                aggregated_features = masked_cls.max(dim=1).values
                no_valid = (mask_bool.sum(dim=1) == 0)
                if no_valid.any():
                    aggregated_features[no_valid] = cls_features[no_valid, 0, :]
            else:
                aggregated_features = torch.max(cls_features, dim=1)[0]
        elif self.aggregation == 'cls_token':
            if mask_bool is not None:
                counts = mask_bool.sum(dim=1)
                first_indices = torch.where(
                    counts > 0,
                    mask_bool.float().argmax(dim=1),
                    torch.zeros_like(counts),
                ).long()
                batch_indices = torch.arange(batch_size, device=x.device)
                aggregated_features = cls_features[batch_indices, first_indices, :]
            else:
                aggregated_features = cls_features[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return aggregated_features

    def get_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Public API to fetch aggregated features without applying the mlp_head."""
        return self._compute_aggregated_features(x, mask=mask)
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"DINOv3_Encoder Parameters:")
        print(f"  • Total: {total:,}")
        print(f"  • Trainable: {trainable:,}")
        print(f"  • Frozen: {total - trainable:,}")
        
        return trainable, total


def load_dinov3_encoder(model_path, **kwargs):
    """
    Helper function to load DINOv3 encoder with error handling
    
    Args:
        model_path: Path to DINOv3 model directory
        **kwargs: Additional arguments for DINOv3_Encoder
        
    Returns:
        DINOv3_Encoder instance
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DINOv3 model path not found: {model_path}")
    
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DINOv3 config.json not found in {model_path}")
    
    model_file = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"DINOv3 model.safetensors not found in {model_path}")
    
    return DINOv3_Encoder(model_path, **kwargs)
