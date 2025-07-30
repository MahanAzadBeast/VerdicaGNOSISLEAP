"""
Multi-Task Model Architecture Demonstration
Shows how both ChemBERTa and Chemprop are designed for multi-task learning on 14 oncoproteins
"""

import modal
import pandas as pd
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=12.0.0"
])

app = modal.App("multitask-demo")
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def demonstrate_multitask_architecture():
    """
    Demonstrate how both ChemBERTa and Chemprop handle multi-task learning
    """
    
    print("🎯 Multi-Task Architecture Demonstration")
    print("=" * 60)
    
    # Load the dataset
    dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
    if not dataset_path.exists():
        return {"error": "Dataset not found"}
    
    df = pd.read_csv(dataset_path)
    target_cols = [col for col in df.columns if col != 'canonical_smiles']
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"🎯 Oncoprotein Targets: {len(target_cols)}")
    print(f"🧬 Unique Compounds: {len(df)}")
    
    print("\n🎯 TARGET COVERAGE ANALYSIS:")
    print("-" * 40)
    
    coverage_data = []
    for target in target_cols:
        non_null = df[target].notna().sum()
        coverage_pct = (non_null / len(df)) * 100
        coverage_data.append({
            'target': target,
            'compounds': non_null,
            'coverage_pct': coverage_pct
        })
        print(f"{target:>8}: {non_null:>4} compounds ({coverage_pct:>5.1f}%)")
    
    total_data_points = sum(item['compounds'] for item in coverage_data)
    print(f"\n📈 TOTAL DATA POINTS: {total_data_points:,}")
    
    print("\n🤖 MULTI-TASK MODEL ARCHITECTURES:")
    print("=" * 60)
    
    print("\n🧠 ChemBERTa Multi-Task Architecture:")
    print("   1. Input: SMILES → Tokenization")
    print("   2. Backbone: Transformer encoder (ChemBERTa)")
    print("   3. Pooling: [CLS] token representation")
    print("   4. Shared Layer: Linear(hidden_size → hidden_size)")
    print("   5. Task Heads: 14 separate Linear(hidden_size → 1) layers")
    print("   6. Loss: Masked MSE (only compute loss where data exists)")
    print("   7. Output: pIC50 predictions for all 14 targets")
    
    print("\n🧪 Chemprop Multi-Task GNN Architecture:")
    print("   1. Input: SMILES → Molecular graph")
    print("   2. Graph Encoder: Message Passing Neural Network")
    print("   3. Node Features: Atom-level representations")
    print("   4. Graph Pooling: Aggregated molecular representation")
    print("   5. FFN: Feed-forward network for each target")
    print("   6. Multi-task Loss: Scaled loss across all targets")
    print("   7. Output: pIC50 predictions for all 14 targets")
    
    print("\n🎯 MULTI-TASK ADVANTAGES:")
    print("-" * 30)
    print("✓ Single model handles all 14 targets")
    print("✓ Shared molecular representations")
    print("✓ Transfer learning between targets") 
    print("✓ Handles missing data gracefully")
    print("✓ More efficient than 14 separate models")
    print("✓ Consistent predictions across targets")
    
    print("\n📊 TRAINING DATA DISTRIBUTION:")
    print("-" * 35)
    
    # Sample some molecules to show multi-target nature
    sample_df = df.head(5)
    
    print("Sample compounds with multi-target data:")
    for idx, row in sample_df.iterrows():
        smiles = row['canonical_smiles']
        targets_with_data = [col for col in target_cols if pd.notna(row[col])]
        print(f"  {smiles[:50]}... → {len(targets_with_data)} targets")
    
    # Show sparsity pattern
    print(f"\n🕳️ DATA SPARSITY:")
    total_possible = len(df) * len(target_cols)
    actual_data = total_data_points
    sparsity = (1 - actual_data / total_possible) * 100
    print(f"   Possible data points: {total_possible:,}")
    print(f"   Actual data points: {actual_data:,}")
    print(f"   Sparsity: {sparsity:.1f}%")
    print(f"   → Multi-task models handle this sparsity elegantly!")
    
    return {
        "dataset_shape": list(df.shape),
        "num_targets": len(target_cols),
        "total_data_points": total_data_points,
        "sparsity_percent": sparsity,
        "target_coverage": coverage_data
    }

@app.local_entrypoint()
def run_demo():
    """Run the multi-task demonstration"""
    result = demonstrate_multitask_architecture.remote()
    
    print("\n🎉 READY FOR MULTI-TASK TRAINING!")
    print("=" * 40)
    print("Both ChemBERTa and Chemprop are configured for multi-task learning")
    print("Run: modal run launch_training.py --model both --epochs 50")
    
    return result

if __name__ == "__main__":
    run_demo()