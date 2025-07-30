# Multi-Task ChemBERTa & Chemprop Training for 14 Oncoproteins

Complete training infrastructure for **multi-task molecular property prediction** using ChemBERTa (transformer) and Chemprop (graph neural network) models. Both models are designed to predict bioactivity (pIC50) for **all 14 oncoprotein targets simultaneously** in a single model.

## 🎯 Multi-Task Architecture

### Both Models Train on ALL 14 Oncoproteins Simultaneously:
- **EGFR, HER2, VEGFR2, BRAF, MET, CDK4, CDK6, ALK, MDM2, STAT3, RRM2, CTNNB1, MYC, PI3KCA**
- **Single model** → **14 predictions** per molecule
- **5,401 data points** across **5,022 unique compounds**
- **Handles missing data** gracefully (92.3% sparsity)

### ChemBERTa Multi-Task:
```
SMILES → Tokenizer → Transformer → [CLS] → Shared Layer → 14 Task Heads
```

### Chemprop Multi-Task GNN:
```
SMILES → Molecular Graph → Message Passing → Graph Pooling → 14 FFN Outputs
```

## 🚀 Quick Start

### 0. Verify Multi-Task Setup
```bash
# See multi-task architecture explanation
modal run demo_multitask.py
```

### 1. Test Both Multi-Task Pipelines
```bash
# Validate dataset and test both training pipelines
modal run test_training_pipeline.py
```

### 2. Train Individual Models
```bash
# Train ChemBERTa only
modal run train_chemberta.py::train_chemberta_multitask

# Train Chemprop only  
modal run train_chemprop.py::train_chemprop_multitask
```

### 3. Train Both Models (Recommended)
```bash
# Launch both models with default settings
modal run launch_training.py

# Custom configuration
modal run launch_training.py --model both --epochs 50 --batch-size 32 --run-name "production-run"
```

## 📊 W&B Integration

All training runs are automatically logged to Weights & Biases:

- **Project**: `veridica-ai-training`
- **Groups**: `chemberta` and `chemprop`
- **Metrics**: Training/validation loss, R², MSE, MAE per target
- **Artifacts**: Trained models, predictions, dataset splits
- **Visualizations**: Prediction plots, target performance comparisons

### Accessing Results
1. Visit [wandb.ai](https://wandb.ai/)
2. Navigate to project: `veridica-ai-training`
3. Compare runs across model types and configurations

## 🏗️ Architecture

### ChemBERTa Pipeline (`train_chemberta.py`)
- **Model**: Multi-task transformer for molecular properties
- **Backbone**: `seyonec/ChemBERTa-zinc-base-v1` 
- **Features**: 
  - SMILES tokenization with attention
  - Multi-task regression heads per target
  - Masked loss for missing data
  - Early stopping and checkpointing

### Chemprop Pipeline (`train_chemprop.py`)
- **Model**: Multi-task graph neural network
- **Features**:
  - Molecular graph representations
  - Message passing neural networks
  - RDKit molecular descriptors
  - Hyperparameter optimization support

## 🎯 Dataset Requirements

The pipelines expect a CSV file with:
- `canonical_smiles`: SMILES strings for molecules
- Target columns: pIC50 values for each protein target
- Missing values: NaN for compounds not tested on specific targets

**Current Dataset**: `oncoprotein_multitask_dataset.csv` (5,022 compounds × 14 targets)

## ⚙️ Configuration Options

### ChemBERTa Parameters
```python
train_chemberta_multitask(
    dataset_name="oncoprotein_multitask_dataset",
    model_name="seyonec/ChemBERTa-zinc-base-v1",
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=10,
    max_length=512,
    dropout=0.1,
    # ... see function for all options
)
```

### Chemprop Parameters
```python
train_chemprop_multitask(
    dataset_name="oncoprotein_multitask_dataset",
    batch_size=50,
    learning_rate=1e-4,
    num_epochs=50,
    hidden_size=300,
    depth=3,
    # ... see function for all options
)
```

## 🔬 Advanced Usage

### Hyperparameter Optimization
```bash
# Run W&B sweep for Chemprop
modal run train_chemprop.py::hyperparameter_sweep_chemprop --n-trials 50
```

### Custom Dataset Training
```bash
# Place your CSV in /vol/datasets/ then:
modal run launch_training.py --dataset your_dataset_name --epochs 30
```

### Prediction on New Data
```python
# ChemBERTa predictions
from train_chemberta import load_and_predict_chemberta
predictions = load_and_predict_chemberta.remote(
    model_path="/vol/models/chemberta_model",
    smiles_list=["CCO", "CC(=O)O"]
)

# Chemprop predictions  
from train_chemprop import predict_chemprop
predictions = predict_chemprop.remote(
    model_path="/vol/models/chemprop_model", 
    smiles_list=["CCO", "CC(=O)O"]
)
```

## 📈 Performance Monitoring

### Key Metrics Tracked
- **Per-target R²**: Coefficient of determination for each protein
- **Per-target MSE/MAE**: Mean squared/absolute error
- **Training/validation loss**: Tracked each epoch
- **Model comparison**: Side-by-side ChemBERTa vs Chemprop

### Artifacts Saved
- **Models**: Complete model checkpoints
- **Predictions**: Test set predictions with true values
- **Plots**: Prediction scatter plots per target
- **Metadata**: Training configuration and dataset info

## 🛠️ Infrastructure

### Modal Configuration
- **GPU**: A100 for training (both pipelines)
- **Memory**: 32GB RAM for large datasets
- **Timeout**: 4 hours max per training job
- **Volumes**: Persistent storage for datasets and models

### W&B Authentication
- Uses Modal secret: `wandb-secret`
- Automatically configured - no manual API key needed

## 🚨 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```bash
   # First run the data extraction:
   modal run full_oncoprotein_extractor.py::extract_full_oncoprotein_dataset
   ```

2. **W&B Authentication Error**
   ```bash
   # Verify Modal secret exists:
   modal secret list
   ```

3. **Out of Memory**
   - Reduce batch size: `--batch-size 8`
   - Reduce sequence length: Set `max_length=256` for ChemBERTa

4. **Training Timeout**
   - Reduce epochs: `--epochs 10`
   - Use early stopping (enabled by default)

### Debug Mode
```bash
# Test with minimal configuration
modal run test_training_pipeline.py
```

## 📝 File Structure

```
modal_training/
├── train_chemberta.py          # ChemBERTa training pipeline
├── train_chemprop.py           # Chemprop training pipeline  
├── launch_training.py          # Unified launcher
├── test_training_pipeline.py   # Testing and validation
├── TRAINING_README.md          # This documentation
└── full_oncoprotein_extractor.py  # Dataset preparation
```

## 🎯 Next Steps

1. **Run Test Suite**: `modal run test_training_pipeline.py`
2. **Launch Training**: `modal run launch_training.py --model both --epochs 50`
3. **Monitor in W&B**: Check project `veridica-ai-training`
4. **Compare Models**: Analyze ChemBERTa vs Chemprop performance
5. **Deploy Best Model**: Use top-performing model for predictions

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Review W&B logs for detailed error messages
3. Run test pipeline to isolate issues
4. Verify dataset format and Modal setup

---

**Ready to train world-class molecular property prediction models!** 🧬🤖