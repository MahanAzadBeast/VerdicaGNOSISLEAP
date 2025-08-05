"""
Comprehensive investigation of Model 2 training failure
Analyzing datasets, processing, and ChemBERTa integration issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def investigate_training_failure():
    """
    Comprehensive analysis of why Model 2 training failed with R² = -0.003
    """
    
    print("🔍 COMPREHENSIVE MODEL 2 FAILURE INVESTIGATION")
    print("=" * 80)
    
    findings = {
        "critical_issues": [],
        "data_quality": {},
        "feature_engineering": {},
        "model_architecture": {},
        "chemberta_integration": {},
        "recommendations": []
    }
    
    # 1. ANALYZE FEATURE EXTRACTION LOGIC
    print("\n1️⃣ ANALYZING FEATURE EXTRACTION")
    print("-" * 50)
    
    # The molecular feature extraction from the training script
    def analyze_molecular_features():
        """Analyze the flawed molecular feature extraction"""
        
        test_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib
            "CCO",  # Ethanol
            "C"     # Methane
        ]
        
        def extract_molecular_features(smiles):
            """Replicated from training script"""
            features = np.zeros(2048)
            
            # Basic chemical features
            features[0] = len(smiles)  # Molecular size
            features[1] = smiles.count('C')  # Carbon count
            features[2] = smiles.count('N')  # Nitrogen count
            features[3] = smiles.count('O')  # Oxygen count
            features[4] = smiles.count('S')  # Sulfur count
            features[5] = smiles.count('=')  # Double bonds
            features[6] = smiles.count('#')  # Triple bonds
            features[7] = smiles.count('c')  # Aromatic carbons
            features[8] = smiles.count('(')  # Branching
            features[9] = smiles.count('[')  # Special atoms
            
            # Fill remaining features with derived values
            for i in range(10, 2048):
                features[i] = np.sin(i * len(smiles) / 100) * 0.1  # Pseudo-features
            
            return features
        
        print("🧬 Molecular Feature Analysis:")
        
        feature_quality_issues = []
        
        for smiles in test_smiles:
            features = extract_molecular_features(smiles)
            
            # Analyze feature distribution
            meaningful_features = features[:10]  # First 10 are real
            pseudo_features = features[10:]      # Rest are pseudo
            
            print(f"\nSMILES: {smiles}")
            print(f"  Meaningful features: {meaningful_features}")
            print(f"  Pseudo features range: {pseudo_features.min():.3f} to {pseudo_features.max():.3f}")
            print(f"  Pseudo feature variance: {pseudo_features.var():.6f}")
            
            # Check for issues
            if np.all(pseudo_features == 0):
                feature_quality_issues.append("All pseudo features are zero")
            if pseudo_features.var() < 1e-6:
                feature_quality_issues.append("Pseudo features have negligible variance")
        
        findings["feature_engineering"]["molecular_issues"] = [
            "Only 10 out of 2048 features are meaningful molecular descriptors",
            "2038 features are pseudo-features generated with sin function",
            "Pseudo-features are deterministic and add noise, not signal",
            "No actual ChemBERTa embeddings are used - contradicts requirement",
            "Basic SMILES character counting is too simplistic",
            "Missing critical molecular properties (MW, LogP, TPSA, etc.)"
        ]
        
        return feature_quality_issues
    
    molecular_issues = analyze_molecular_features()
    
    # 2. ANALYZE CELL LINE FEATURES
    print("\n2️⃣ ANALYZING CELL LINE FEATURES")
    print("-" * 50)
    
    def analyze_cell_line_features():
        """Analyze the cell line feature generation"""
        
        test_cell_lines = ["A549", "MCF7", "HCT116", "PC-3"]
        
        print("🦠 Cell Line Feature Analysis:")
        
        cell_line_issues = []
        
        for cell_line in test_cell_lines:
            # Replicated from training script
            features = np.random.RandomState(hash(cell_line) % 2**32).normal(0, 1, 100)
            
            print(f"\nCell line: {cell_line}")
            print(f"  Feature mean: {features.mean():.3f}")
            print(f"  Feature std: {features.std():.3f}")
            print(f"  Feature range: {features.min():.3f} to {features.max():.3f}")
            
            if abs(features.mean()) > 0.1:
                cell_line_issues.append(f"{cell_line} has biased mean")
        
        findings["feature_engineering"]["cell_line_issues"] = [
            "Cell line features are completely random noise",
            "No real genomic data (mutations, CNVs, expression) used",
            "Random features have no correlation with actual cell line sensitivity",
            "100 random features per cell line create false complexity",
            "Deterministic random seed means same features always generated"
        ]
        
        return cell_line_issues
    
    cell_line_issues = analyze_cell_line_features()
    
    # 3. ANALYZE MODEL ARCHITECTURE
    print("\n3️⃣ ANALYZING MODEL ARCHITECTURE")
    print("-" * 50)
    
    def analyze_model_architecture():
        """Analyze the neural network architecture"""
        
        print("🧠 Model Architecture Analysis:")
        
        # Architecture from training script
        molecular_dim = 2048
        cell_line_features = 100  
        hidden_dims = [512, 256, 128]
        
        print(f"Input dimensions:")
        print(f"  Molecular features: {molecular_dim}")
        print(f"  Cell line features: {cell_line_features}")
        print(f"  Total input features: {molecular_dim + cell_line_features}")
        
        print(f"\nArchitecture:")
        print(f"  Molecular encoder: {molecular_dim} → {hidden_dims[0]} → {hidden_dims[1]}")
        print(f"  Cell line encoder: {cell_line_features} → {hidden_dims[1]} → {hidden_dims[2]}")
        print(f"  Combined features: {hidden_dims[1] + hidden_dims[2]} = {hidden_dims[1] + hidden_dims[2]}")
        print(f"  Prediction head: {hidden_dims[1] + hidden_dims[2]} → {hidden_dims[2]} → 64 → 1")
        
        # Calculate total parameters
        params = (molecular_dim * hidden_dims[0] + hidden_dims[0] * hidden_dims[1] +  # molecular encoder
                 cell_line_features * hidden_dims[1] + hidden_dims[1] * hidden_dims[2] +  # cell line encoder  
                 (hidden_dims[1] + hidden_dims[2]) * hidden_dims[2] + hidden_dims[2] * 64 + 64 * 1)  # prediction head
        
        print(f"\nApproximate parameters: {params:,}")
        
        findings["model_architecture"]["issues"] = [
            "Model is overly complex for the quality of input features",
            "1.3M parameters training on noise features leads to overfitting",
            "High dropout rates (0.3, 0.2, 0.1) suggest awareness of overfitting",
            "No regularization techniques beyond dropout",
            "Architecture assumes meaningful features, but inputs are mostly noise"
        ]
    
    analyze_model_architecture()
    
    # 4. CHEMBERTA INTEGRATION ANALYSIS
    print("\n4️⃣ ANALYZING CHEMBERTA INTEGRATION")
    print("-" * 50)
    
    def analyze_chemberta_integration():
        """Analyze ChemBERTa integration (or lack thereof)"""
        
        print("🤖 ChemBERTa Integration Analysis:")
        
        findings["chemberta_integration"]["critical_findings"] = [
            "🚨 MAJOR ISSUE: No ChemBERTa embeddings are actually used!",
            "Training script mentions ChemBERTa in description but uses character counting",
            "ChemBERTa would provide 768-dimensional meaningful embeddings",
            "Current approach: 10 character counts + 2038 sine-wave pseudo features",
            "Missing transformers/ChemBERTa imports in training script",
            "No tokenization or embedding extraction pipeline",
            "No pre-trained model loading or fine-tuning"
        ]
        
        print("❌ ChemBERTa Integration Status: COMPLETELY MISSING")
        print("   - No transformers library usage")
        print("   - No pre-trained model loading") 
        print("   - No SMILES tokenization")
        print("   - No embedding extraction")
        print("   - Using primitive character counting instead")
        
    analyze_chemberta_integration()
    
    # 5. DATA QUALITY ANALYSIS
    print("\n5️⃣ ANALYZING DATA QUALITY")
    print("-" * 50)
    
    def analyze_data_quality():
        """Analyze training data quality issues"""
        
        print("📊 Data Quality Analysis:")
        
        # From training logs, we know:
        training_stats = {
            "primary_dataset": 55100,
            "cleaned_dataset": 53998,
            "ic50_range": "0.031 - 32.755 μM",
            "log_ic50_range": "-1.506 - 1.515",
            "training_samples": 43198,
            "validation_samples": 10800,
            "unique_cell_lines": 1000
        }
        
        print("From training logs:")
        for key, value in training_stats.items():
            print(f"  {key}: {value}")
        
        findings["data_quality"]["issues"] = [
            "Dataset source unknown - not using actual GDSC data effectively",
            "1000 unique cell lines seems unrealistically high for GDSC",
            "IC50 range looks reasonable but data source questionable",
            "Training/validation split seems standard (80/20)",
            "Data preprocessing removes only 2% of records - may be too lenient"
        ]
        
    analyze_data_quality()
    
    # 6. TRAINING PROCESS ANALYSIS
    print("\n6️⃣ ANALYZING TRAINING PROCESS")
    print("-" * 50)
    
    # From training logs
    training_progression = [
        {"epoch": 1, "train_loss": 0.4220, "val_loss": 0.3648, "val_r2": -0.0147},
        {"epoch": 2, "train_loss": 0.3758, "val_loss": 0.3619, "val_r2": -0.0066},
        {"epoch": 3, "train_loss": 0.3712, "val_loss": 0.3606, "val_r2": -0.0031},
        {"epoch": 18, "final": True, "val_r2": -0.0031}
    ]
    
    print("Training progression analysis:")
    for epoch_data in training_progression:
        if epoch_data.get("final"):
            print(f"  Final (epoch {epoch_data['epoch']}): R² = {epoch_data['val_r2']}")
        else:
            print(f"  Epoch {epoch_data['epoch']}: Loss = {epoch_data['train_loss']:.4f} → {epoch_data['val_loss']:.4f}, R² = {epoch_data['val_r2']}")
    
    print("\n🔍 Training Pattern Analysis:")
    print("  - Loss decreases but R² stays negative")
    print("  - Model learns to minimize MSE but can't predict better than mean")
    print("  - Classic sign of learning noise patterns instead of real relationships")
    print("  - Early stopping at epoch 18 due to no improvement")
    
    # 7. CRITICAL FINDINGS SUMMARY
    print("\n🚨 CRITICAL FINDINGS SUMMARY")
    print("=" * 80)
    
    findings["critical_issues"] = [
        "🔴 FEATURE ENGINEERING FAILURE: 99.5% of molecular features are meaningless pseudo-features",
        "🔴 NO CHEMBERTA: Despite claims, no ChemBERTa embeddings are used - only character counting",
        "🔴 RANDOM CELL LINE FEATURES: Cell line features are pure random noise, not genomic data",
        "🔴 SIGNAL-TO-NOISE RATIO: Model tries to learn from ~10 real features + 2138 noise features",
        "🔴 ARCHITECTURE MISMATCH: Complex 1.3M parameter model on mostly noise data",
        "🔴 DATA SOURCE UNCLEAR: Training on unknown dataset instead of verified GDSC data"
    ]
    
    # 8. RECOMMENDATIONS
    findings["recommendations"] = [
        "🔧 IMPLEMENT REAL CHEMBERTA: Use actual ChemBERTa embeddings for molecular features",
        "🔧 USE REAL GENOMIC DATA: Replace random features with actual mutation/expression data", 
        "🔧 SIMPLIFY ARCHITECTURE: Start with simpler model given current data quality",
        "🔧 VERIFY DATA SOURCE: Ensure training on real GDSC cancer cell line data",
        "🔧 FEATURE VALIDATION: Validate that features correlate with known drug-target relationships",
        "🔧 BASELINE COMPARISON: Implement simple linear regression baseline for comparison"
    ]
    
    print("\n💡 IMMEDIATE NEXT STEPS:")
    for i, rec in enumerate(findings["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Save findings
    timestamp = datetime.now().isoformat()
    findings["analysis_date"] = timestamp
    
    with open("/app/modal_training/model2_failure_analysis.json", "w") as f:
        json.dump(findings, f, indent=2)
    
    print(f"\n💾 Analysis saved to: /app/modal_training/model2_failure_analysis.json")
    print("\n🎯 CONCLUSION: Model 2 failed because it's training on noise, not real molecular/genomic features!")
    
    return findings

if __name__ == "__main__":
    findings = investigate_training_failure()