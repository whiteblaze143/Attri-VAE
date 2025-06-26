# Introduction

Myocardial Infarction (MI), commonly known as a heart attack, remains a leading cause of mortality worldwide. Early prediction of complications following MI is crucial for timely intervention and improved patient outcomes. However, this task presents significant challenges due to the complex, temporal nature of patient data and the need for clinically interpretable predictions.

Traditional machine learning approaches in healthcare often treat patient data as static snapshots, ignoring the crucial temporal progression of clinical parameters. Furthermore, many models operate as "black boxes," providing predictions without clear clinical justification, which limits their adoption in medical practice.

In this work, we present a novel Temporal Progressive Attribute-Regularized Variational Autoencoder (Attri-VAE) for MI complication prediction. Our approach addresses several key challenges:

1. **Temporal Integrity**: We preserve the temporal progression of patient data from admission through 72 hours post-MI, capturing the dynamic nature of clinical parameters.

2. **Clinical Interpretability**: By aligning latent dimensions with key clinical attributes (age, blood pressure, potassium levels, etc.), we ensure that model predictions are grounded in medically meaningful features.

3. **Medical Safety**: We implement physiologically plausible constraints to prevent unrealistic predictions and ensure clinical validity.

4. **Progressive Learning**: Our model learns to weigh later timepoints more heavily, reflecting the increasing importance of recent clinical data in complication prediction.

The key contributions of this work include:

- A novel VAE architecture that explicitly models temporal progression while maintaining clinical interpretability
- Integration of medical domain knowledge through attribute regularization and safety constraints
- Comprehensive evaluation using both traditional ML metrics and clinically relevant measures
- Detailed analysis of model interpretability and its implications for clinical decision-making

Our results demonstrate that the Attri-VAE achieves superior performance compared to traditional approaches while providing clinically meaningful insights into the prediction process. This work represents a significant step toward developing machine learning models that are both accurate and interpretable in medical applications. 