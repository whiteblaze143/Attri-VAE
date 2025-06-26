import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import logging
import sys
import random
from tqdm import tqdm
import subprocess

from model.mi_complications_vae import MIComplicationsVAE
from model.train_mi_vae import MIComplicationsTrainer
from model.mi_dataset import load_mi_data, create_data_loaders

def setup_logging(output_dir):
    """Set up logging configuration"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def setup_tensorboard(output_dir):
    """Set up TensorBoard for tracking metrics"""
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=tensorboard_dir)

def launch_tensorboard(logdir, port=6006):
    """Launch TensorBoard server"""
    cmd = f"tensorboard --logdir={logdir} --port={port}"
    try:
        process = subprocess.Popen(cmd, shell=True)
        print(f"TensorBoard started on http://localhost:{port}")
        return process
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")
        return None

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_config(config, output_dir):
    """Save training configuration"""
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def log_metrics_to_tensorboard(writer, metrics, step, prefix=''):
    """Log metrics to TensorBoard"""
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f"{prefix}/{k}", v, step)

class TensorboardCallback:
    """Callback for logging metrics to TensorBoard during training"""
    def __init__(self, writer):
        self.writer = writer
        self.step = 0
        
    def on_epoch_end(self, epoch, train_metrics, val_metrics):
        """Log metrics at the end of each epoch"""
        log_metrics_to_tensorboard(self.writer, train_metrics, epoch, prefix='train')
        log_metrics_to_tensorboard(self.writer, val_metrics, epoch, prefix='val')
        
        # Log learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'lr/group_{i}', param_group['lr'], epoch)
        
    def set_optimizer(self, optimizer):
        """Set the optimizer for learning rate tracking"""
        self.optimizer = optimizer

def main():
    # Configuration
    config = {
        # Model parameters
        "latent_dim": 32,
        "embed_dim": 16,
        "n_filters_ENC": (8, 16, 32, 64, 2),
        "n_filters_DEC": (64, 32, 16, 8, 4, 2),
        
        # Training parameters
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "n_epochs": 100,
        "early_stopping_patience": 10,
        
        # Loss weights
        "beta": 1.0,  # KL loss weight
        "gamma": 0.1,  # Attribute regularization weight
        "delta": 0.1,  # Medical safety weight
        
        # Data parameters
        "val_size": 0.2,
        "test_size": 0.1,
        "random_state": 42,
        
        # TensorBoard parameters
        "tensorboard_port": 6006,
        "launch_tensorboard": True,
        "add_model_graph": False  # Set to False to avoid graph visualization errors
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"mi_vae_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training in {output_dir}")
    
    # Set up TensorBoard
    writer = setup_tensorboard(output_dir)
    logger.info(f"TensorBoard logs will be saved to {output_dir / 'tensorboard'}")
    
    # Launch TensorBoard if requested
    tensorboard_process = None
    if config["launch_tensorboard"]:
        tensorboard_process = launch_tensorboard(
            output_dir / "tensorboard", 
            port=config["tensorboard_port"]
        )
        if tensorboard_process:
            logger.info(f"TensorBoard started on http://localhost:{config['tensorboard_port']}")
        else:
            logger.warning("Failed to start TensorBoard automatically")
    
    # Save configuration
    save_config(config, output_dir)
    
    # Set random seed
    set_seed(config["random_state"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_timepoints, y, attributes = load_mi_data()
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            X_timepoints,
            y,
            attributes,
            batch_size=config["batch_size"],
            val_size=config["val_size"],
            test_size=config["test_size"],
            random_state=config["random_state"]
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model = MIComplicationsVAE(
            input_dims=[X.shape[1] for X in X_timepoints.values()],
            latent_dim=config["latent_dim"],
            embed_dim=config["embed_dim"],
            attribute_dims={attr: i for i, attr in enumerate(attributes.columns)},
            medical_safety=True,
            n_filters_ENC=config["n_filters_ENC"],
            n_filters_DEC=config["n_filters_DEC"]
        ).to(device)
        
        # Add model graph to TensorBoard (optional)
        if config["add_model_graph"]:
            try:
                sample_input = [torch.FloatTensor(X.iloc[:1].values).to(device) for X in X_timepoints.values()]
                sample_attrs = torch.FloatTensor(attributes.iloc[:1].values).to(device)
                writer.add_graph(model, (sample_input, sample_attrs))
                logger.info("Added model graph to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to add model graph to TensorBoard: {e}")
        
        # Initialize optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Initialize learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create TensorBoard callback
        tb_callback = TensorboardCallback(writer)
        tb_callback.set_optimizer(optimizer)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = MIComplicationsTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            beta=config["beta"],
            gamma=config["gamma"],
            delta=config["delta"],
            callbacks=[tb_callback]  # Pass TensorBoard callback to trainer
        )
        
        # Train model
        logger.info("Starting training...")
        best_model = trainer.train(
            n_epochs=config["n_epochs"],
            early_stopping_patience=config["early_stopping_patience"]
        )
        
        # Save final model
        model_path = output_dir / "final_model.pth"
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, model_path)
        logger.info(f"Saved final model to {model_path}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.validate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Log test metrics to TensorBoard
        log_metrics_to_tensorboard(writer, test_metrics, 0, prefix='test')
        
        # Save test metrics
        metrics_path = output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        logger.info("Training completed successfully!")
        
        # Close TensorBoard writer
        writer.close()
        
        # Print TensorBoard access instructions
        logger.info("\nTo view training metrics in TensorBoard:")
        logger.info(f"1. Run: tensorboard --logdir={output_dir}/tensorboard")
        logger.info(f"2. Open http://localhost:6006 in your browser")
        
        # Clean up TensorBoard process if it was started
        if tensorboard_process:
            logger.info("Stopping TensorBoard server...")
            tensorboard_process.terminate()
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        
        # Clean up TensorBoard process if it was started
        if tensorboard_process:
            tensorboard_process.terminate()
        
        raise

if __name__ == "__main__":
    main() 