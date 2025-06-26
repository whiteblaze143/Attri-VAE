import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MedicalSafetyLayer(nn.Module):
    """
    Layer to enforce physiological ranges for outputs
    """
    def __init__(self, feature_ranges=None):
        """
        Initialize layer with physiological ranges
        
        Args:
            feature_ranges: Dictionary mapping feature names to (min, max) range tuples
        """
        super(MedicalSafetyLayer, self).__init__()
        self.feature_ranges = feature_ranges or {
            # Systolic BP (mmHg)
            'S_AD': (80, 180),
            # Diastolic BP (mmHg)
            'D_AD': (40, 120),
            # Heart rate (bpm)
            'RATE_AD': (40, 180),
            # Body temperature (°C)
            'TEMP_AD': (35, 41),
            # Potassium (mmol/L)
            'K_BLOOD': (3.0, 6.0),
            # Sodium (mmol/L)
            'Na_BLOOD': (130, 150),
            # White blood cells (10^9/L)
            'L_BLOOD': (4.0, 25.0)
        }
        
    def forward(self, x, feature_names=None):
        """
        Apply safety constraints to outputs
        
        Args:
            x: Input tensor
            feature_names: List of feature names corresponding to dimensions
            
        Returns:
            x_safe: Output tensor with values within safe ranges
        """
        # If feature names not provided, return unchanged
        if feature_names is None:
            return x, None
        
        # Initialize loss
        safety_loss = 0.0
        
        # Apply safety constraints
        for i, name in enumerate(feature_names):
            for key, (min_val, max_val) in self.feature_ranges.items():
                if key in name:
                    # Calculate penalty for out-of-range values
                    below_min = F.relu(min_val - x[:, i])
                    above_max = F.relu(x[:, i] - max_val)
                    penalty = below_min + above_max
                    safety_loss += torch.mean(penalty)
        
        return x, safety_loss

class MIComplicationsVAE(nn.Module):
    """
    VAE for MI complications prediction with temporal progression and attribute regularization
    """
    def __init__(
        self,
        input_dims,
        latent_dim=32,
        embed_dim=16,
        attribute_dims=None,
        medical_safety=True,
        n_filters_ENC=(8, 16, 32, 64, 2),
        n_filters_DEC=(64, 32, 16, 8, 4, 2)
    ):
        """
        Initialize model
        
        Args:
            input_dims: List of input dimensions for each timepoint
            latent_dim: Dimension of the latent space
            embed_dim: Dimension of time embeddings
            attribute_dims: Dictionary mapping attribute name to latent dimension
            medical_safety: Whether to enforce medical safety constraints
            n_filters_ENC: Number of filters in encoder
            n_filters_DEC: Number of filters in decoder
        """
        super(MIComplicationsVAE, self).__init__()
        
        self.input_dims = input_dims if isinstance(input_dims, list) else list(input_dims)
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.attribute_dims = attribute_dims or {}
        self.medical_safety = medical_safety
        
        # Time embeddings (one for each timepoint)
        self.time_embeddings = nn.Embedding(len(input_dims), embed_dim)
        
        # Encoders for each timepoint
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + embed_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ) for dim in input_dims
        ])
        
        # Latent projectors (mu and logvar)
        self.mu_projector = nn.Linear(32 * len(input_dims), latent_dim)
        self.logvar_projector = nn.Linear(32 * len(input_dims), latent_dim)
        
        # Attention mechanism for temporal progression
        self.attn_weights = nn.Parameter(torch.ones(len(input_dims)) / len(input_dims))
        
        # Integration module
        self.integration = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Attribute predictor
        self.attribute_predictor = nn.Linear(latent_dim, len(self.attribute_dims))
        
        # Classifier for complications prediction
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        # Decoders for each timepoint
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + embed_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, dim)
            ) for dim in input_dims
        ])
        
        # Medical safety layer
        if medical_safety:
            self.safety_layer = MedicalSafetyLayer()
    
    def encode(self, x_list, attributes=None):
        """
        Encode a list of inputs from different timepoints
        
        Args:
            x_list: List of input tensors
            attributes: Clinical attributes tensor
            
        Returns:
            mu: Mean vector
            logvar: Log variance vector
        """
        batch_size = x_list[0].size(0)
        encodings = []
        
        # Generate time embeddings
        time_indices = torch.arange(len(self.encoders), device=x_list[0].device)
        time_embeds = self.time_embeddings(time_indices)
        
        # Encode each timepoint with its time embedding
        for i, (x, encoder) in enumerate(zip(x_list, self.encoders)):
            # Add time embedding to input
            time_embed = time_embeds[i].expand(batch_size, -1)
            x_t = torch.cat([x, time_embed], dim=1)
            
            # Encode
            h = encoder(x_t)
            encodings.append(h)
        
        # Concatenate all encodings
        concat_encoding = torch.cat(encodings, dim=1)
        
        # Project to latent space
        mu = self.mu_projector(concat_encoding)
        logvar = self.logvar_projector(concat_encoding)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from distribution
        
        Args:
            mu: Mean vector
            logvar: Log variance vector
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent representation
        
        Args:
            z: Latent vector
            
        Returns:
            recon_list: List of reconstructions for each timepoint
            safety_loss: Medical safety loss if enabled
        """
        batch_size = z.size(0)
        recon_list = []
        safety_loss = None
        
        # Generate time embeddings
        time_indices = torch.arange(len(self.decoders), device=z.device)
        time_embeds = self.time_embeddings(time_indices)
        
        for i, decoder in enumerate(self.decoders):
            # Add time embedding to latent vector
            time_embed = time_embeds[i].expand(batch_size, -1)
            z_t = torch.cat([z, time_embed], dim=1)
            
            # Decode
            x_recon = decoder(z_t)
            
            # Apply medical safety constraints if enabled
            if self.medical_safety:
                x_recon, time_safety_loss = self.safety_layer(x_recon)
                if time_safety_loss is not None:
                    if safety_loss is None:
                        safety_loss = time_safety_loss
                    else:
                        safety_loss += time_safety_loss
            
            recon_list.append(x_recon)
        
        return recon_list, safety_loss
    
    def predict_attributes(self, z):
        """
        Predict attributes from latent vector
        
        Args:
            z: Latent vector
            
        Returns:
            attr_pred: Predicted attributes
        """
        return self.attribute_predictor(z)
    
    def forward(self, x_list, attributes=None):
        """
        Forward pass
        
        Args:
            x_list: List of input tensors
            attributes: Clinical attributes tensor
            
        Returns:
            recon_list: List of reconstructions
            mu: Mean vector
            logvar: Log variance vector
            attr_pred: Predicted attributes
            y_pred: Predicted complication probability
            safety_loss: Medical safety loss if enabled
        """
        # Encode
        mu, logvar = self.encode(x_list, attributes)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Integrate latent representation
        z = self.integration(z)
        
        # Predict attributes
        attr_pred = self.predict_attributes(z)
        
        # Predict complications
        y_pred = self.classifier(z)
        
        # Decode
        recon_list, safety_loss = self.decode(z)
        
        return recon_list, mu, logvar, attr_pred, y_pred, safety_loss

    def compute_losses(self, outputs, targets, attributes):
        # Unpack outputs
        reconstructions, mu, logvar, pred_attributes, y_pred, safety_loss = outputs
        x_list, y, _ = targets
        
        # 1. Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(x_list, reconstructions)
        
        # 2. KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. Attribute regularization loss (from Attri-VAE)
        attr_loss = self.mse_loss(pred_attributes, attributes)
        
        # 4. Classification loss for complications
        cls_loss = self.bce_loss(y_pred, y.view(-1, 1))
        
        # 5. Medical safety loss (your existing implementation)
        
        # Total loss with weighting
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * attr_loss + cls_loss
        if safety_loss is not None:
            total_loss += self.delta * safety_loss
        
        return total_loss, {...}  # Return individual losses for logging 