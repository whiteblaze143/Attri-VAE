# Methodology

## Data Preprocessing and Temporal Handling

Our approach begins with careful preprocessing of the UCI MI complications dataset. We implement a temporal-aware preprocessing pipeline that:

1. **Timepoint Definition**: We define four critical timepoints: admission, 24h, 48h, and 72h post-MI. Each timepoint contains a specific subset of features available at that stage.

2. **Clinical Feature Engineering**: We derive clinically meaningful features such as:
   - Pulse pressure (difference between systolic and diastolic blood pressure)
   - Age-based risk groups
   - Laboratory threshold indicators
   - Time-to-hospital risk groups

3. **Temporal Data Preprocessing**: We implement a specialized `TemporalDataPreprocessor` that:
   - Maintains temporal relationships between features
   - Handles missing values using iterative imputation
   - Scales numerical features while preserving temporal patterns
   - Encodes categorical variables appropriately

## Model Architecture

The core of our approach is the Temporal Progressive Attri-VAE, which consists of several key components:

1. **Time Embeddings**: We learn time-specific embeddings for each timepoint, allowing the model to capture temporal context.

2. **Progressive Encoders**: Each timepoint has its own encoder that:
   - Processes the input features
   - Incorporates time embeddings
   - Produces latent representations

3. **Attribute Regularization**: We align latent dimensions with clinical attributes through:
   - Explicit mapping of clinical parameters to latent dimensions
   - Regularization terms that enforce clinical plausibility
   - Attention mechanisms that weight timepoints appropriately

4. **Medical Safety Layer**: We implement constraints that:
   - Enforce physiologically plausible ranges for outputs
   - Prevent unrealistic predictions
   - Ensure clinical validity

## Loss Functions

Our model employs a comprehensive loss function that combines:

1. **Reconstruction Loss**: Ensures accurate reconstruction of input features while preserving temporal relationships.

2. **KL Divergence Loss**: Regularizes the latent space to follow a standard normal distribution.

3. **Classification Loss**: Optimizes for accurate complication prediction.

4. **Attribute Regularization Loss**: Aligns latent dimensions with clinical attributes.

5. **Progression Loss**: Encourages appropriate weighting of later timepoints.

6. **Clinical Plausibility Loss**: Ensures medically valid predictions.

## Training and Evaluation

We implement a rigorous training and evaluation pipeline:

1. **Cross-Validation**: We use a specialized `TemporalCVSplitter` that:
   - Preserves temporal structure
   - Prevents data leakage between timepoints
   - Ensures proper validation

2. **Hyperparameter Optimization**: We use Optuna to optimize:
   - Latent dimension size
   - Embedding dimension
   - Loss function weights
   - Learning rates

3. **Evaluation Metrics**: We assess model performance using:
   - Traditional metrics (AUC, accuracy, F1-score)
   - Clinical metrics (sensitivity, specificity)
   - Calibration metrics (Brier score)
   - Interpretability measures

## Implementation Details

The model is implemented in PyTorch with the following key features:

1. **Modular Architecture**: Each component is implemented as a separate module for clarity and maintainability.

2. **GPU Acceleration**: The model supports both CPU and GPU training.

3. **Reproducibility**: We implement seed setting and deterministic operations.

4. **Logging and Visualization**: Comprehensive logging of training progress and results.

The code is structured to be both efficient and interpretable, with detailed documentation and type hints throughout. 