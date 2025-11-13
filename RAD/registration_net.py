import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint
import datetime

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        # Level 0: Combined initial downsampling and feature expansion (1/2)
        self.level0_conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.level0_norm1 = nn.InstanceNorm3d(16)
        self.level0_conv2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, padding_mode='replicate')
        self.level0_norm2 = nn.InstanceNorm3d(16)
        self.level0_conv3 = nn.Conv3d(16, 16, kernel_size=3, padding=1, padding_mode='replicate')
        self.level0_norm3 = nn.InstanceNorm3d(16)
        
        # Level 1: Moderate feature expansion - Downsampling (1/4)
        self.level1_conv1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.level1_norm1 = nn.InstanceNorm3d(32)  
        self.level1_conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode='replicate')
        self.level1_norm2 = nn.InstanceNorm3d(32)
        
        # Level 2: Intermediate expansion without downsampling
        self.level2_conv1 = nn.Conv3d(32, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.level2_norm1 = nn.InstanceNorm3d(64)
        self.level2_conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.level2_norm2 = nn.InstanceNorm3d(64)
        
        # Level 3: Balanced growth - Downsampling (1/8)
        self.level3_conv1 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.level3_norm1 = nn.InstanceNorm3d(128)
        self.level3_conv2 = nn.Conv3d(128, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.level3_norm2 = nn.InstanceNorm3d(128)
        
        # Level 4: Deep feature extraction - Downsampling (1/16)
        self.level4_conv1 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.level4_norm1 = nn.InstanceNorm3d(256)
        self.level4_conv2 = nn.Conv3d(256, 256, kernel_size=3, padding=1, padding_mode='replicate')
        self.level4_norm2 = nn.InstanceNorm3d(256)
                
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # Level 0 - Combined downsampling and expansion with residual
        x = self.level0_conv1(x)
        x = self.level0_norm1(x)
        x = self.leaky_relu(x)
        x = self.level0_conv2(x)
        x = self.level0_norm2(x)
        x = self.leaky_relu(x)
        x = self.level0_conv2(x)
        x = self.level0_norm2(x)
        x = self.leaky_relu(x)
        res0 = self.level0_conv3(x)
        res0 = self.level0_norm3(res0)
        feat0 = self.leaky_relu(x + res0)
        
        # Level 1 with residual
        feat1 = self.level1_conv1(feat0)
        feat1 = self.level1_norm1(feat1)
        feat1 = self.leaky_relu(feat1)
        feat1 = self.level1_conv2(feat1)
        feat1 = self.level1_norm1(feat1)
        feat1 = self.leaky_relu(feat1)
        res1 = self.level1_conv2(feat1)
        res1 = self.level1_norm2(res1)
        feat1 = self.leaky_relu(feat1 + res1)
        
        # Level 2 with residual
        feat2 = self.level2_conv1(feat1)
        feat2 = self.level2_norm1(feat2)
        feat2 = self.leaky_relu(feat2)
        feat2 = self.level2_conv2(feat2)
        feat2 = self.level2_norm1(feat2)
        feat2 = self.leaky_relu(feat2)
        res2 = self.level2_conv2(feat2)
        res2 = self.level2_norm2(res2)
        feat2 = self.leaky_relu(feat2 + res2)
        
        # Level 3 with residual
        feat3 = self.level3_conv1(feat2)
        feat3 = self.level3_norm1(feat3)
        feat3 = self.leaky_relu(feat3)
        feat3 = self.level3_conv2(feat3)
        feat3 = self.level3_norm1(feat3)
        feat3 = self.leaky_relu(feat3)
        res3 = self.level3_conv2(feat3)
        res3 = self.level3_norm2(res3)
        feat3 = self.leaky_relu(feat3 + res3)
        
        # Level 4 with residual
        feat4 = self.level4_conv1(feat3)
        feat4 = self.level4_norm1(feat4)
        feat4 = self.leaky_relu(feat4)
        feat4 = self.level4_conv2(feat4)
        feat4 = self.level4_norm1(feat4)
        feat4 = self.leaky_relu(feat4)
        res4 = self.level4_conv2(feat4)
        res4 = self.level4_norm2(res4)
        feat4 = self.leaky_relu(feat4 + res4)
                      
        return {"level1": feat1, "level2": feat2, "level3": feat3, "level4": feat4}

class FeatureDelta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Feature dimension
        feature_dim = 32
        
        # global context extraction
        self.fixed_global_extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(4),  # Reduce to 4x4x4 volume (memory efficient)
                nn.Flatten(),
                nn.Linear(64, feature_dim)
            ) for _ in range(channels)
        ])
        
        self.moving_global_extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(4),
                nn.Flatten(),
                nn.Linear(64, feature_dim)
            ) for _ in range(channels)
        ])
        
        # Cross-modal fusion
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
                nn.LeakyReLU(inplace=True)
            ) for _ in range(channels)
        ])
        
        # Transform predictors
        self.transform_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 6)  # 3 rotation + 3 translation
            ) for _ in range(channels)
        ])
    
    def forward(self, fixed_features, moving_features):
        B, C = fixed_features.shape[:2]
        device = fixed_features.device
        
        translations = torch.empty(B, C * 3, device=device)
        rotations = torch.empty(B, C * 3, device=device)
        
        for c in range(C):
            # Extract global features directly
            fixed_c = fixed_features[:, c:c+1]
            moving_c = moving_features[:, c:c+1]
            
            # Extract global features from each modality
            fixed_global = self.fixed_global_extractors[c](fixed_c)
            moving_global = self.moving_global_extractors[c](moving_c)
            
            # Concatenate for cross-modal fusion
            combined = torch.cat([fixed_global, moving_global], dim=1)
            
            # Apply fusion layers (MLP instead of transformer)
            fused_features = self.fusion_layers[c](combined)
            
            # Predict transform
            transform = self.transform_predictors[c](fused_features)
            
            # Split into rotation and translation
            rotations[:, c*3:(c+1)*3] = transform[:, :3]
            translations[:, c*3:(c+1)*3] = transform[:, 3:6]
            
        return translations, rotations
    
class TransformationHeads(nn.Module):
    """
    Predicts rigid transformation parameters (rotation and translation) with constraints.
    Outputs transformation parameters in LPS coordinate system:
    - Rotation: [rx, ry, rz] in radians, constrained to ±45 degrees
      - rx: rotation around L/R axis
      - ry: rotation around P/A axis
      - rz: rotation around S/I axis
    - Translation: [tx, ty, tz] in voxel units, constrained to ±15% of image dimensions
      - tx: translation along L/R axis
      - ty: translation along P/A axis
      - tz: translation along S/I axis
    """
    def __init__(self, prog_channels=1440, max_rotation_degrees=45.0):
        super().__init__()
        self.translation_net = nn.Linear(prog_channels, 3)
        self.rotation_net = nn.Linear(prog_channels, 3)
        
        # Convert degrees to radians
        self.max_rotation = max_rotation_degrees * (torch.pi / 180.0)
        
    def forward(self, prog_trans, prog_rot, image_dimensions=None):
        # Calculate max translation based on image dimensions (15% of each dimension)
        if image_dimensions is None:
            # Default fallback if dimensions not provided
            max_translation = torch.tensor([10.0, 10.0, 10.0], device=prog_trans.device)
        else:
            max_translation = torch.tensor(image_dimensions, device=prog_trans.device) * 0.15
        
        # Apply linear layers
        raw_rotation = self.rotation_net(prog_rot)
        raw_translation = self.translation_net(prog_trans)
        
        # Apply tanh to constrain outputs to specified ranges
        # tanh outputs values in [-1, 1], which we scale todesired ranges
        rotation = torch.tanh(raw_rotation) * self.max_rotation
        translation = torch.tanh(raw_translation) * max_translation.unsqueeze(0)
        
        return rotation, translation
        
class RegistrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Progressive pathway with updated extractor
        self.progressive_extractor = FeatureExtractor()
        
        # Delta modules for each level
        self.progressive_delta_1 = FeatureDelta(channels=32)    # Level 1
        self.progressive_delta_2 = FeatureDelta(channels=64)    # Level 2
        self.progressive_delta_3 = FeatureDelta(channels=128)   # Level 3
        self.progressive_delta_4 = FeatureDelta(channels=256)   # Level 4
                
        # Total channels: (32+64+128+256)*3 = 1,440
        self.transform_heads = TransformationHeads(prog_channels=1440)
        
        self.to(self.device)
        self._init_weights()

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        if fixed.dim() == 4:
            fixed = fixed.unsqueeze(1)
        if moving.dim() == 4:
            moving = moving.unsqueeze(1)
            
        fixed = fixed.to(self.device)
        moving = moving.to(self.device)
        
        # Extract image dimensions from the fixed image
        _, _, D, H, W = fixed.shape
        image_dimensions = [W, H, D]

        # Progressive pathway
        fixed_prog = checkpoint(self.progressive_extractor, fixed, use_reentrant=False)
        moving_prog = checkpoint(self.progressive_extractor, moving, use_reentrant=False)
        
        # Compute deltas at each level (level0 excluded)
        translations = []
        rotations = []
        
        # Process Level 4 (deepest level)
        fixed_feat = fixed_prog["level4"]
        moving_feat = moving_prog["level4"]
        adaptive_pool = nn.AdaptiveAvgPool3d(fixed_feat.shape[2:])
        moving_feat = adaptive_pool(moving_feat)
        trans4, rot4 = self.progressive_delta_4(fixed_feat, moving_feat)
        translations.append(trans4)
        rotations.append(rot4)
        del fixed_feat, moving_feat, adaptive_pool, trans4, rot4
        
        # Process Level 3
        fixed_feat = fixed_prog["level3"]
        moving_feat = moving_prog["level3"]
        adaptive_pool = nn.AdaptiveAvgPool3d(fixed_feat.shape[2:])
        moving_feat = adaptive_pool(moving_feat)
        trans3, rot3 = self.progressive_delta_3(fixed_feat, moving_feat)
        translations.append(trans3)
        rotations.append(rot3)
        del fixed_feat, moving_feat, adaptive_pool, trans3, rot3
        
        # Process Level 2
        fixed_feat = fixed_prog["level2"]
        moving_feat = moving_prog["level2"]
        adaptive_pool = nn.AdaptiveAvgPool3d(fixed_feat.shape[2:])
        moving_feat = adaptive_pool(moving_feat)
        trans2, rot2 = self.progressive_delta_2(fixed_feat, moving_feat)
        translations.append(trans2)
        rotations.append(rot2)
        del fixed_feat, moving_feat, adaptive_pool, trans2, rot2
     
        # Process Level 1
        fixed_feat = fixed_prog["level1"]
        moving_feat = moving_prog["level1"]
        adaptive_pool = nn.AdaptiveAvgPool3d(fixed_feat.shape[2:])
        moving_feat = adaptive_pool(moving_feat)
        trans1, rot1 = self.progressive_delta_1(fixed_feat, moving_feat)
        translations.append(trans1)
        rotations.append(rot1)
        del fixed_feat, moving_feat, adaptive_pool, trans1, rot1
        
        # Combine features from all levels (excluding level0)
        trans_prog = torch.cat(translations, dim=1)
        rot_prog = torch.cat(rotations, dim=1)

        # Predict transformation with dimension-aware constraints
        rotation, translation = self.transform_heads(trans_prog, rot_prog, image_dimensions)
        
        return torch.cat([rotation, translation], dim=1)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if (m is self.transform_heads.rotation_net or 
                    m is self.transform_heads.translation_net):  
                    # Initialize final layer weights to be very close to zero
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    def load_model(self, filename: str, device: Optional[str] = None) -> Optional[dict]:
        """Load model with backward compatibility support."""
        checkpoint = torch.load(filename, map_location=device if device else self.device)
        
        # Handle different saving formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                # Legacy format where the dict is directly the state_dict
                self.load_state_dict(checkpoint)
        else:
            # Direct state_dict format
            self.load_state_dict(checkpoint)
            
        if device:
            self.device = device
            self.to(device)
        
        # Return metadata if available
        return checkpoint.get('metadata', None) if isinstance(checkpoint, dict) else None
            
    def save_model(self, filename: str, metadata: dict = None) -> None:
        """Save model with enhanced metadata."""
        # Calculate model size estimate in MB
        model_size = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)
        
        # Build standard format dictionary
        save_dict = {
            'state_dict': self.state_dict(),
            'metadata': {
                # Architecture information
                'parameter_count': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'model_size_mb': model_size,
                'feature_dimensions': {
                    'level1': 32,
                    'level2': 64,
                    'level3': 128,
                    'level4': 256,
                },
                # System information
                'timestamp': datetime.datetime.now().isoformat(),
                'torch_version': torch.__version__,
            }
        }
        
        # Add custom metadata if provided
        if metadata:
            save_dict['metadata'].update(metadata)
            
        torch.save(save_dict, filename)

class TransformLoss(nn.Module):
    """Loss for rigid transform using standard MSE
        With SITK transform format [rx, ry, rz, tx, ty, tz] in LPS coordinate system
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are of right shape
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
                
        rot_loss = F.mse_loss(pred[:, 0:3], target[:, 0:3])
        trans_loss = F.mse_loss(pred[:, 3:6], target[:, 3:6])
        
        return trans_loss + 150 * rot_loss 