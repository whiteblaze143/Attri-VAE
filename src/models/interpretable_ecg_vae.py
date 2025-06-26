import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class ECGGroupEncoder(nn.Module):
    """Encoder for ECG signal group"""
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

class ClinicalEncoder(nn.Module):
    """Encoder for clinical features group"""
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

class InterpretableECGVAE(nn.Module):
    """Interpretable VAE for ECG data with group structure and clinical attribute regularization"""
    
    def __init__(
        self,
        ecg_input_dim: int,
        clinical_input_dim: int,
        latent_dim: int,
        ecg_hidden_dims: List[int] = [256, 128],
        clinical_hidden_dims: List[int] = [128, 64],
        attribute_dims: Dict[str, int] = None,
        beta: float = 1.0,
        gamma: float = 0.1
    ):
        super().__init__()
        
        # Group-specific encoders
        self.ecg_encoder = ECGGroupEncoder(ecg_input_dim, ecg_hidden_dims, latent_dim)
        self.clinical_encoder = ClinicalEncoder(clinical_input_dim, clinical_hidden_dims, latent_dim)
        
        # Decoders
        self.ecg_decoder = nn.Sequential(
            nn.Linear(latent_dim, ecg_hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(ecg_hidden_dims[-1], ecg_input_dim)
        )
        
        self.clinical_decoder = nn.Sequential(
            nn.Linear(latent_dim, clinical_hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(clinical_hidden_dims[-1], clinical_input_dim)
        )
        
        # Attribute regularization
        self.attribute_dims = attribute_dims or {}
        self.attribute_predictors = nn.ModuleDict({
            attr: nn.Linear(latent_dim, dim)
            for attr, dim in attribute_dims.items()
        })
        
        self.beta = beta  # KL divergence weight
        self.gamma = gamma  # Attribute regularization weight
        
    def encode(self, ecg: torch.Tensor, clinical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode both ECG and clinical data"""
        ecg_mu, ecg_log_var = self.ecg_encoder(ecg)
        clinical_mu, clinical_log_var = self.clinical_encoder(clinical)
        
        # Combine group-specific encodings
        mu = (ecg_mu + clinical_mu) / 2
        log_var = (ecg_log_var + clinical_log_var) / 2
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent representation"""
        ecg_recon = self.ecg_decoder(z)
        clinical_recon = self.clinical_decoder(z)
        return ecg_recon, clinical_recon
    
    def predict_attributes(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict clinical attributes from latent space"""
        return {
            attr: predictor(z)
            for attr, predictor in self.attribute_predictors.items()
        }
    
    def forward(
        self,
        ecg: torch.Tensor,
        clinical: torch.Tensor,
        attributes: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional attribute supervision"""
        mu, log_var = self.encode(ecg, clinical)
        z = self.reparameterize(mu, log_var)
        ecg_recon, clinical_recon = self.decode(z)
        
        # Compute losses
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        recon_loss = F.mse_loss(ecg_recon, ecg) + F.mse_loss(clinical_recon, clinical)
        
        # Attribute prediction and regularization
        attr_loss = 0
        if attributes is not None:
            pred_attributes = self.predict_attributes(z)
            for attr, pred in pred_attributes.items():
                if attr in attributes:
                    attr_loss += F.mse_loss(pred, attributes[attr])
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * attr_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'attr_loss': attr_loss,
            'ecg_recon': ecg_recon,
            'clinical_recon': clinical_recon,
            'z': z
        }
    
    def get_attention_maps(self, ecg: torch.Tensor) -> torch.Tensor:
        """Get attention maps for ECG signal regions"""
        # Implementation depends on specific attention mechanism
        # This is a placeholder for the actual implementation
        return torch.ones_like(ecg)
    
    def get_feature_importance(self, clinical: torch.Tensor) -> torch.Tensor:
        """Get feature importance scores for clinical parameters"""
        # Implementation depends on specific feature importance method
        # This is a placeholder for the actual implementation
        return torch.ones(clinical.shape[1]) 