# Interpretable ECG VAE: Myocardial Infarction Prediction with Explainable AI

An interpretable Variational Autoencoder for ECG signal analysis and myocardial infarction prediction, combining unsupervised representation learning with clinical interpretability.

## 🔬 Overview

This project implements an interpretable VAE architecture for ECG data analysis with the following key features:

- **Group-based encoding**: Separate encoders for ECG signals and clinical features
- **Attribute regularization**: Clinical attribute prediction for interpretability  
- **Attention mechanisms**: Attribute-wise attention maps for explainability
- **Myocardial infarction prediction**: Binary classification with uncertainty quantification

## 🏗️ Architecture

The model consists of:
- **ECG Group Encoder**: Processes ECG signal features
- **Clinical Encoder**: Handles demographic and clinical parameters
- **Shared Latent Space**: Combined representation for reconstruction and prediction
- **Attribute Predictors**: Clinical attribute prediction heads
- **Attention Module**: Generates interpretable attention maps

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── interpretable_ecg_vae.py
│   │   └── best_temporal_attrivae.pth
│   ├── data/                     # Data processing utilities
│   ├── utils/                    # Helper functions
│   ├── attribute_wise_attention.py
│   ├── training_main.py
│   ├── training_testing_funcs.py
│   └── testing_acdc.py
├── data_features/                # Feature extraction and preprocessing
├── notebooks/                    # Jupyter notebooks
│   └── Final_Project.ipynb      # Main analysis notebook
├── papers/                       # Research papers and documentation
│   ├── Final Paper.pdf
│   ├── unsupervised_learning_paper.tex
│   └── refs.bib
├── docs/                         # Additional documentation
├── output/                       # Training outputs
├── mi_prediction_output/         # MI prediction results
├── results/                      # Experimental results
└── requirements.txt              # Dependencies
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 1.8+
CUDA (optional, for GPU acceleration)
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Final_Project
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Data Preprocessing**:
```bash
python data_features/data_load_feature_extract.py
```

2. **Training**:
```bash
python train.py
```

3. **Evaluation**:
```bash
python main.py
```

4. **Jupyter Analysis**:
```bash
jupyter notebook Final_Project.ipynb
```

## 🔧 Configuration

Key hyperparameters can be modified in the training scripts:

- `latent_dim`: Latent space dimensionality (default: 128)
- `beta`: KL divergence weight (default: 1.0) 
- `gamma`: Attribute regularization weight (default: 0.1)
- `learning_rate`: Optimizer learning rate (default: 1e-3)

## 📊 Results

The model achieves:
- **Reconstruction Quality**: Low MSE on ECG signal reconstruction
- **MI Prediction**: High accuracy with interpretable predictions
- **Clinical Correlation**: Strong correlation with known MI risk factors
- **Attention Maps**: Clinically relevant ECG region highlighting

Detailed results and analysis are available in:
- `Final_Project.ipynb` - Interactive analysis
- `papers/Final Paper.pdf` - Complete research paper
- `mi_prediction_output/` - Raw experimental results

## 🎯 Key Features

### Interpretability
- **Attribute-wise attention**: Highlights ECG regions relevant to specific clinical attributes
- **Feature importance**: Quantifies contribution of different clinical parameters
- **Latent space visualization**: t-SNE/UMAP plots of learned representations

### Clinical Integration
- **Multi-modal input**: Combines ECG signals with clinical parameters
- **Risk stratification**: Probability scores for MI prediction
- **Uncertainty quantification**: Confidence intervals for predictions

## 📖 Usage Examples

### Training a New Model
```python
from src.models.interpretable_ecg_vae import InterpretableECGVAE

model = InterpretableECGVAE(
    ecg_input_dim=500,
    clinical_input_dim=10,
    latent_dim=128,
    attribute_dims={'age': 1, 'sex': 1}
)
```

### Generating Attention Maps
```python
from src.attribute_wise_attention import attribute_wise_attn

attention_generator = attribute_wise_attn(model, target_layer='encoder.conv3')
attention_maps = attention_generator.generate()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{interpretable_ecg_vae_2024,
  title={Interpretable ECG VAE for Myocardial Infarction Prediction},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- Original Attri-VAE paper: [arXiv:2203.10417](http://arxiv.org/abs/2203.10417)
- PTB-XL ECG Database
- Additional references in `papers/refs.bib`

## 📞 Contact

For questions and support, please open an issue or contact [maintainer email].

---

**Keywords**: ECG Analysis, Variational Autoencoders, Interpretable AI, Myocardial Infarction, Attention Mechanisms, Healthcare AI
