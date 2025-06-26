import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import copy
from collections import defaultdict

class MIComplicationsTrainer:
    """
    Trainer for MI Complications VAE model
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-4,
        weight_decay=1e-5,
        beta=1.0,
        gamma=0.1,
        delta=0.1,
        callbacks=None
    ):
        """
        Initialize trainer
        
        Args:
            model: MIComplicationsVAE model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use (cuda or cpu)
            lr: Learning rate
            weight_decay: Weight decay
            beta: KL divergence weight
            gamma: Attribute regularization weight
            delta: Medical safety weight
            callbacks: List of callbacks for training events
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss weights
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # MSE loss for reconstruction
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # BCE loss for classification
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Set optimizer in callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'set_optimizer'):
                callback.set_optimizer(self.optimizer)
    
    def compute_losses(self, outputs, targets, attributes):
        """
        Compute all losses for training
        
        Args:
            outputs: Model outputs
            targets: Target values
            attributes: Clinical attributes
            
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Unpack outputs
        reconstructions, mu, logvar, pred_attributes, y_pred, safety_loss = outputs
        x_list, y, _ = targets
        
        # Ensure y and y_pred have the same shape
        y = y.view(-1, 1) if y_pred.shape[-1] == 1 else y
        
        # Reconstruction loss (MSE)
        recon_loss = 0
        for i, (x, recon) in enumerate(zip(x_list, reconstructions)):
            recon_loss += self.mse_loss(recon, x)
        recon_loss /= len(x_list)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Attribute regularization loss
        attr_loss = self.mse_loss(pred_attributes, attributes)
        
        # Classification loss
        cls_loss = self.bce_loss(y_pred, y)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * attr_loss + cls_loss
        
        # Add medical safety loss if available
        if safety_loss is not None:
            total_loss += self.delta * safety_loss
        
        # Return individual losses for logging
        loss_dict = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'attr_loss': attr_loss.item(),
            'cls_loss': cls_loss.item(),
        }
        
        if safety_loss is not None:
            loss_dict['safety_loss'] = safety_loss.item()
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch):
        """
        Train model for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
            metrics: Dictionary of metrics
        """
        self.model.train()
        running_loss = 0.0
        loss_dict_sum = defaultdict(float)
        predictions = []
        targets = []
        
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (x_list, y, attributes) in enumerate(pbar):
            # Move data to device
            x_list = [x.to(self.device) for x in x_list]
            y = y.to(self.device)
            attributes = attributes.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x_list, attributes)
            
            # Compute loss
            loss, batch_loss_dict = self.compute_losses(outputs, (x_list, y, attributes), attributes)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            for k, v in batch_loss_dict.items():
                loss_dict_sum[k] += v
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Store predictions for metrics
            y_pred = outputs[4]
            predictions.append(y_pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
        
        # Concatenate predictions and targets
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Reshape for metrics calculation
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        
        # Calculate metrics
        metrics = {}
        for k, v in loss_dict_sum.items():
            metrics[k] = v / len(self.train_loader)
        
        # Calculate classification metrics
        try:
            metrics['auroc'] = roc_auc_score(targets, predictions)
            metrics['auprc'] = average_precision_score(targets, predictions)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
        
        return metrics['loss'], metrics
    
    def validate(self, dataloader=None):
        """
        Validate model on validation set
        
        Args:
            dataloader: DataLoader to use for validation (default: self.val_loader)
            
        Returns:
            avg_loss: Average loss for validation
            metrics: Dictionary of metrics
        """
        if dataloader is None:
            dataloader = self.val_loader
            
        self.model.eval()
        running_loss = 0.0
        loss_dict_sum = defaultdict(float)
        predictions = []
        targets = []
        
        with torch.no_grad():
            # Create progress bar
            pbar = tqdm(dataloader, desc='Validation')
            
            for batch_idx, (x_list, y, attributes) in enumerate(pbar):
                # Move data to device
                x_list = [x.to(self.device) for x in x_list]
                y = y.to(self.device)
                attributes = attributes.to(self.device)
                
                # Forward pass
                outputs = self.model(x_list, attributes)
                
                # Compute loss
                loss, batch_loss_dict = self.compute_losses(outputs, (x_list, y, attributes), attributes)
                
                # Update running loss
                running_loss += loss.item()
                for k, v in batch_loss_dict.items():
                    loss_dict_sum[k] += v
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                # Store predictions for metrics
                y_pred = outputs[4]
                predictions.append(y_pred.detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())
        
        # Concatenate predictions and targets
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Reshape for metrics calculation
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        
        # Calculate metrics
        metrics = {}
        for k, v in loss_dict_sum.items():
            metrics[k] = v / len(dataloader)
        
        # Calculate classification metrics
        try:
            metrics['auroc'] = roc_auc_score(targets, predictions)
            metrics['auprc'] = average_precision_score(targets, predictions)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
        
        return metrics
    
    def train(self, n_epochs=100, early_stopping_patience=10):
        """
        Train model for n_epochs
        
        Args:
            n_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            
        Returns:
            best_model: Model with best validation loss
        """
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train AUROC: {train_metrics['auroc']:.4f}, Val AUROC: {val_metrics['auroc']:.4f}")
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Check if model improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Call callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        return best_model 