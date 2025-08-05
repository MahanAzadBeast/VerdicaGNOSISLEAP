"""
Diagnose why Model 2 R² is low (0.0003) and create improvement roadmap
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """Analyze the quality of our training data"""
    
    print("🔍 DIAGNOSING LOW R² PERFORMANCE")
    print("=" * 60)
    
    # 1. LOAD AND EXAMINE GDSC DATA QUALITY
    print("\n1️⃣ DATA QUALITY ANALYSIS")
    
    # Check what datasets we actually have locally
    local_data_files = [
        "/app/datasets/cytotoxicity_data.csv",
        "/app/datasets/bulk_tox21_cytotoxicity_data.csv"
    ]
    
    for file_path in local_data_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            print(f"\n📁 {Path(file_path).name}:")
            print(f"   Records: {len(df):,}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Analyze IC50 distribution if present
            ic50_cols = [col for col in df.columns if 'ic50' in col.lower()]
            if ic50_cols:
                ic50_col = ic50_cols[0]
                ic50_values = pd.to_numeric(df[ic50_col], errors='coerce')
                ic50_clean = ic50_values.dropna()
                
                print(f"   IC50 column: {ic50_col}")
                print(f"   IC50 range: {ic50_clean.min():.3f} - {ic50_clean.max():.3f}")
                print(f"   IC50 median: {ic50_clean.median():.3f}")
                print(f"   Missing values: {ic50_values.isna().sum():,} ({ic50_values.isna().mean()*100:.1f}%)")
                
                # Check for unrealistic values
                unrealistic = ic50_clean[(ic50_clean < 0.001) | (ic50_clean > 1000)]
                print(f"   Unrealistic IC50s (<0.001 or >1000): {len(unrealistic):,}")
                
            # Check SMILES quality
            smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
            if smiles_cols:
                smiles_col = smiles_cols[0]
                smiles_data = df[smiles_col].dropna()
                print(f"   SMILES column: {smiles_col}")
                print(f"   Valid SMILES: {len(smiles_data):,}")
                print(f"   Unique molecules: {smiles_data.nunique():,}")
                
                # Sample SMILES for quality check
                print(f"   Sample SMILES: {smiles_data.iloc[0][:50]}...")
                
            # Check cell line diversity
            cell_cols = [col for col in df.columns if 'cell' in col.lower() or 'line' in col.lower()]
            if cell_cols:
                cell_col = cell_cols[0]
                cell_lines = df[cell_col].dropna()
                print(f"   Cell line column: {cell_col}")
                print(f"   Unique cell lines: {cell_lines.nunique():,}")
                print(f"   Top cell lines: {cell_lines.value_counts().head(3).index.tolist()}")
    
    # 2. ANALYZE FEATURE-TARGET CORRELATIONS
    print("\n2️⃣ FEATURE-TARGET CORRELATION ANALYSIS")
    
    # Load our current model's feature extractors
    try:
        import sys
        sys.path.append('/app/backend')
        from model2_cytotoxicity_predictor import ProductionMolecularEncoder, RealGenomicFeatureExtractor
        
        # Test feature extraction on sample molecules
        mol_encoder = ProductionMolecularEncoder()
        genomic_extractor = RealGenomicFeatureExtractor()
        
        # Sample molecules with known activity patterns
        test_molecules = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Imatinib", "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"),
            ("Cisplatin", "[Pt](Cl)(Cl)(N)N"),
            ("5-Fluorouracil", "FC1=CNC(=O)NC1=O"),
            ("Paclitaxel", "CC(=O)O[C@H]1C[C@@]2(C[C@@H]([C@H]3[C@]4(CO[C@@H]4C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)C)OC(=O)C")
        ]
        
        print("\n🧬 Molecular Feature Analysis:")
        for name, smiles in test_molecules:
            try:
                features = mol_encoder.encode_smiles(smiles)
                print(f"   {name}: {len(features)} features extracted")
                print(f"     MW≈{features[0]:.1f}, LogP≈{features[1]:.2f}, TPSA≈{features[4]:.1f}")
            except Exception as e:
                print(f"   {name}: Error - {e}")
        
        print("\n🦠 Genomic Feature Analysis:")
        test_cell_lines = ["A549", "MCF7", "HCT116", "PC-3"]
        for cell_line in test_cell_lines:
            features = genomic_extractor.extract_features(cell_line)
            print(f"   {cell_line}: {len(features)} genomic features extracted")
            
    except Exception as e:
        print(f"   Feature extraction error: {e}")
    
    # 3. IDENTIFY IMPROVEMENT OPPORTUNITIES
    print("\n3️⃣ IMPROVEMENT OPPORTUNITIES IDENTIFIED")
    
    improvement_plan = {
        "data_quality": [
            "✓ Use high-quality GDSC1/GDSC2 datasets (>500K records)",
            "✓ Filter IC50 values to realistic range (0.01-100 μM)",
            "✓ Remove duplicate compound-cell line pairs",
            "✓ Ensure proper IC50 units and scaling"
        ],
        "molecular_features": [
            "🔥 Implement REAL ChemBERTa embeddings (768-dim)",
            "✓ Add molecular fingerprints (ECFP4, MACCS)",
            "✓ Include 3D molecular descriptors",
            "✓ Add drug-likeness properties (QED, SA_Score)"
        ],
        "genomic_features": [
            "🔥 Use ACTUAL GDSC genomic data from CCLE",
            "✓ Real mutation status from genomics databases",
            "✓ Gene expression profiles (RNA-seq)",
            "✓ Copy number variations (CNV) data",
            "✓ Pathway activity scores from GSEA"
        ],
        "model_architecture": [
            "✓ Implement attention mechanism for feature importance",
            "✓ Use ensemble methods (RF + NN + XGBoost)",
            "✓ Multi-task learning (predict multiple endpoints)",
            "✓ Transfer learning from pre-trained models"
        ],
        "training_strategy": [
            "✓ Cross-validation instead of single split",
            "✓ Hyperparameter optimization (Optuna/Ray Tune)",
            "✓ Data augmentation techniques",
            "✓ Advanced regularization (dropout schedules, weight decay)"
        ]
    }
    
    for category, items in improvement_plan.items():
        print(f"\n🎯 {category.replace('_', ' ').title()}:")
        for item in items:
            print(f"   {item}")
    
    return improvement_plan

def benchmark_existing_approaches():
    """Benchmark against known successful approaches"""
    
    print("\n4️⃣ BENCHMARK AGAINST SUCCESSFUL APPROACHES")
    
    # Known successful drug-cell line prediction methods
    successful_approaches = {
        "DeepDR": {"R²": 0.73, "method": "Deep learning with drug structure + cell line genomics"},
        "GraphDRP": {"R²": 0.68, "method": "Graph neural networks for drugs + genomics"},
        "DeepCDR": {"R²": 0.71, "method": "CNN for drug images + genomic features"},
        "MOLI": {"R²": 0.65, "method": "Multi-omics integration"},
        "tCNNS": {"R²": 0.69, "method": "Transformer + CNN architecture"}
    }
    
    print("🏆 State-of-art methods achieving R² > 0.6:")
    for method, info in successful_approaches.items():
        print(f"   {method}: R² = {info['R²']:.2f} ({info['method']})")
    
    print(f"\n📊 Our current performance: R² = 0.0003")
    print(f"🎯 Target improvement needed: {0.6 - 0.0003:.4f} R² increase")
    
    return successful_approaches

if __name__ == "__main__":
    improvement_plan = analyze_data_quality()
    successful_approaches = benchmark_existing_approaches()
    
    print("\n" + "="*60)
    print("🚀 NEXT STEPS TO ACHIEVE R² > 0.6:")
    print("   1. Implement ChemBERTa embeddings")
    print("   2. Use real GDSC genomic data") 
    print("   3. Expand to 500K+ high-quality records")
    print("   4. Advanced model architectures")
    print("   5. Ensemble methods")
    print("="*60)