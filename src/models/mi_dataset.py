import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MIDataset(Dataset):
    """
    Custom dataset for MI complications data with temporal structure
    """
    def __init__(
        self,
        X_timepoints,
        y,
        attributes,
        transform=None,
        target_transform=None
    ):
        """
        Initialize dataset
        
        Args:
            X_timepoints: Dictionary of feature dataframes for each timepoint
            y: Target array
            attributes: Clinical attributes for regularization
            transform: Optional transform to be applied to features
            target_transform: Optional transform to be applied to targets
        """
        self.X_timepoints = X_timepoints
        self.y = y
        self.attributes = attributes
        self.transform = transform
        self.target_transform = target_transform
        
        # Convert to tensors
        self.X_tensors = {
            tp: torch.FloatTensor(X.values) 
            for tp, X in X_timepoints.items()
        }
        self.y_tensor = torch.FloatTensor(y.values)
        self.attributes_tensor = torch.FloatTensor(attributes.values)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Get features for each timepoint
        x_list = [self.X_tensors[tp][idx] for tp in sorted(self.X_tensors.keys())]
        
        # Get target and attributes
        y = self.y_tensor[idx]
        attributes = self.attributes_tensor[idx]
        
        # Apply transforms if specified
        if self.transform:
            x_list = [self.transform(x) for x in x_list]
        if self.target_transform:
            y = self.target_transform(y)
        
        return x_list, y, attributes

def create_data_loaders(
    X_timepoints,
    y,
    attributes,
    batch_size=32,
    val_size=0.2,
    test_size=0.1,
    random_state=42
):
    """
    Create train, validation, and test data loaders
    
    Args:
        X_timepoints: Dictionary of feature dataframes for each timepoint
        y: Target array
        attributes: Clinical attributes for regularization
        batch_size: Batch size for data loaders
        val_size: Proportion of data to use for validation
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    # First split into train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Then split train+val into train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size/(1-test_size),
        random_state=random_state,
        stratify=y.iloc[train_val_idx]
    )
    
    # Create datasets
    train_dataset = MIDataset(
        {tp: X.iloc[train_idx] for tp, X in X_timepoints.items()},
        y.iloc[train_idx],
        attributes.iloc[train_idx]
    )
    
    val_dataset = MIDataset(
        {tp: X.iloc[val_idx] for tp, X in X_timepoints.items()},
        y.iloc[val_idx],
        attributes.iloc[val_idx]
    )
    
    test_dataset = MIDataset(
        {tp: X.iloc[test_idx] for tp, X in X_timepoints.items()},
        y.iloc[test_idx],
        attributes.iloc[test_idx]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader

def load_mi_data():
    """
    Load and preprocess MI complications dataset
    
    Returns:
        X_timepoints: Dictionary of feature dataframes for each timepoint
        y: Target array
        attributes: Clinical attributes for regularization
    """
    from ucimlrepo import fetch_ucirepo
    
    # Fetch the UCI MI complications dataset
    mi = fetch_ucirepo(id=579)
    X_full = mi.data.features
    y_full = mi.data.targets
    
    # Checking if dataset is loaded properly
    if X_full is None or X_full.empty:
        raise ValueError("Failed to load features from UCI MI complications dataset")
    
    # Create any-complication binary target
    y = pd.Series((y_full.sum(axis=1) > 0).astype(int))
    
    # Define timepoints and their corresponding features
    timepoints = {
        'admission': [col for col in X_full.columns if '_ADM' in col or col in ['AGE', 'SEX', 'TIME_B_S']],
        '24h': [col for col in X_full.columns if '_24_' in col or '_24H' in col],
        '48h': [col for col in X_full.columns if '_48_' in col or '_48H' in col],
        '72h': [col for col in X_full.columns if '_72_' in col or '_72H' in col]
    }
    
    # Create feature sets for each timepoint
    X_timepoints = {}
    for tp, cols in timepoints.items():
        if cols:  # Only process if we have columns for this timepoint
            X_tp = X_full[cols].copy()
            X_timepoints[tp] = X_tp
    
    # Select clinical attributes for regularization
    # These should be features that are clinically meaningful
    attribute_cols = [
        'AGE',
        'S_AD_ORIT',
        'D_AD_ORIT',
        'K_BLOOD',
        'L_BLOOD',
        'TIME_B_S'
    ]
    
    # Filter to only include columns that exist in the dataset
    attribute_cols = [col for col in attribute_cols if col in X_full.columns]
    attributes = X_full[attribute_cols].copy()
    
    # Handle missing values
    for tp, X in X_timepoints.items():
        # Check if any columns are fully NaN
        empty_cols = X.columns[X.isna().all()].tolist()
        if empty_cols:
            print(f"Warning: Dropping columns with all NaN values in {tp}: {empty_cols}")
            X.drop(columns=empty_cols, inplace=True)
            
        # Now fill remaining NaNs with column means
        X.fillna(X.mean(), inplace=True)
    
    attributes.fillna(attributes.mean(), inplace=True)
    
    # Scale features
    scalers = {}
    for tp, X in X_timepoints.items():
        if not X.empty:  # Only scale if we have data
            scaler = StandardScaler()
            X_timepoints[tp] = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            scalers[tp] = scaler
    
    # Scale attributes
    if not attributes.empty:
        attr_scaler = StandardScaler()
        attributes = pd.DataFrame(
            attr_scaler.fit_transform(attributes),
            columns=attributes.columns,
            index=attributes.index
        )
    
    return X_timepoints, y, attributes 