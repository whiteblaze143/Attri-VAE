# Attri-VAE: Interpretable Variational Autoencoder with Group Sparsity

A deep learning approach for unsupervised analysis of myocardial infarction complication patterns using VAEs with group sparsity regularization.

## Project Overview

This project implements a variational autoencoder with group sparsity regularization to learn interpretable latent representations of patient data. The model enhances clinical interpretability while preserving predictive power for analyzing myocardial infarction complications.

## Features

- VAE architecture with attribute-level and group-level sparsity regularization
- Handles missing values common in clinical datasets
- Provides interpretable latent dimensions aligned with clinical concepts
- Visualizations of latent space and clinical correlations
- Comparative analysis with baseline methods

## Repository Structure
Attri-VAE/
├── README.md
├── requirements.txt
├── data/ # Data directory (see data/README.md for dataset instructions)
├── notebooks/ # Jupyter notebooks for experiments and analysis
├── src/ # Source code
│ ├── data/ # Data loading and preprocessing
│ ├── models/ # Model implementations
│ └── utils/ # Utility functions and visualization
└── tests/ # Unit tests
## Installation

```bash
# Clone the repository
git clone https://github.com/whiteblaze143/Attri-VAE.git
cd Attri-VAE

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Notebook

```bash
jupyter notebook notebooks/Final_Project.ipynb
```

## Dataset

This project uses the Myocardial Infarction Complications dataset from the UCI Machine Learning Repository.

## Citation

If you use this code in your research, please cite:

@misc{manivannan2024attrivae,
author = {Manivannan, Mithun},
title = {Attri-VAE: Interpretable Variational Autoencoder with Group Sparsity},
year = {2024},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/whiteblaze143/Attri-VAE}}
}