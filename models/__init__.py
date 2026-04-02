# ProtFunc Enhanced Models
# Backward compatible with existing ResidualMLP while adding attention pooling capabilities

from .enhanced_mlp import EnhancedResidualMLP, AttentionPooling, FocalLoss, MultiTaskHead

__all__ = ['EnhancedResidualMLP', 'AttentionPooling', 'FocalLoss', 'MultiTaskHead']
