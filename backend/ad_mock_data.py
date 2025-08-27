"""
Mock Training Data Builder for AD Layer Testing

This script creates realistic mock training data for testing the Applicability Domain layer.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_mock_training_data(n_compounds: int = 1000, 
                               n_targets: int = 10,
                               save_path: str = "/app/backend/ad_cache/mock_training_data.csv") -> pd.DataFrame:
    """
    Generate mock training data for AD layer testing.
    
    Args:
        n_compounds: Number of unique compounds
        n_targets: Number of targets
        save_path: Path to save the data
        
    Returns:
        DataFrame with training data
    """
    logger.info(f"Generating mock training data: {n_compounds} compounds, {n_targets} targets")
    
    # Define realistic targets based on Gnosis I target list
    targets = ['EGFR', 'BRAF', 'CDK2', 'PARP1', 'ALK', 'MET', 'JAK2', 'PLK1', 'AURKA', 'MTOR'][:n_targets]
    
    # Drug-like SMILES templates for realistic molecules (UPDATED with real drug scaffolds)
    smiles_templates = [
        # Real kinase inhibitor scaffolds
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib-like
        "Nc1ncnc2c1cnn2",  # Purine scaffold 
        "Nc1nc(N)nc(N)c1N",  # Triaminopyrimidine
        "c1ccc2nc(N)nc(N)c2c1",  # Quinazoline
        "COc1cc2ncnc(N)c2cc1OC",  # Quinazoline with methoxy
        "Fc1ccc(cc1)C(=O)Nc2nc(N)nc(N)c2",  # Fluorinated kinase inhibitor
        "c1ccc(cc1)C(=O)Nc2ncnc3c2cnc(N)n3",  # Complex purine derivative
        # PARP inhibitor scaffolds
        "NC(=O)c1ccc2c(c1)oc(=O)n2",  # Olaparib-like
        "NC(=O)c1ccc(cc1)CN2CCC(CC2)",  # Benzamide scaffold
        "c1ccc(cc1)C(=O)NC2CCCCC2",  # Cyclohexyl benzamide
        # GPCR ligand scaffolds
        "c1ccc2c(c1)nccc2N",  # Quinoline amine
        "CN(C)CCc1ccc2c(c1)cccc2O",  # Naphthalene derivative
        "c1ccc(cc1)OCCN(C)C",  # Phenoxyethyl amine
        # Original simple scaffolds (keep some for diversity)
        "c1ccccc1",  # Benzene ring
        "c1cccnc1",  # Pyridine
        "c1ccncc1",  # Pyrimidine
        "C1CCC(CC1)N",  # Cyclohexylamine
        "C1CCNCC1",    # Piperidine
    ]
    
    # Generate compounds
    compounds_data = []
    
    for i in range(n_compounds):
        try:
            # Create random drug-like molecule
            base_template = np.random.choice(smiles_templates)
            
            # Add random modifications
            modifications = []
            
            # Add substituents
            if np.random.random() < 0.7:  # 70% chance of substituent
                substituents = ["C", "CC", "CCC", "F", "Cl", "Br", "CF3", "OH", "OC", "N", "NC"]
                modifications.append(np.random.choice(substituents))
            
            if np.random.random() < 0.5:  # 50% chance of second substituent
                substituents = ["C", "F", "Cl", "OH", "N"]
                modifications.append(np.random.choice(substituents))
            
            # Create SMILES (simplified - in practice would use proper synthesis)
            if base_template == "c1ccccc1" and modifications:
                if len(modifications) == 1:
                    smiles = f"c1ccc(cc1){modifications[0]}"
                elif len(modifications) == 2:
                    smiles = f"c1cc(c(cc1){modifications[0]}){modifications[1]}"
                else:
                    smiles = base_template
            else:
                smiles = base_template
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                smiles = base_template  # Fallback to simple template
                mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # Canonicalize
                smiles = Chem.MolToSmiles(mol, canonical=True)
                
                compounds_data.append({
                    'ligand_id': f'COMPOUND_{i+1:04d}',
                    'smiles': smiles
                })
                
        except Exception as e:
            logger.warning(f"Error generating compound {i}: {e}")
            # Fallback to simple benzene
            compounds_data.append({
                'ligand_id': f'COMPOUND_{i+1:04d}',
                'smiles': 'c1ccccc1'
            })
    
    logger.info(f"Generated {len(compounds_data)} valid compounds")
    
    # Create training records (compound-target-assay combinations)
    training_data = []
    
    for compound in compounds_data:
        # Each compound tested against random subset of targets
        n_targets_tested = np.random.randint(1, min(6, len(targets) + 1))
        tested_targets = np.random.choice(targets, n_targets_tested, replace=False)
        
        for target in tested_targets:
            # Each target can have multiple assay types
            assay_types = ['Binding_IC50', 'Functional_IC50', 'EC50']
            
            # Higher probability for IC50 assays
            assay_probs = [0.6, 0.3, 0.1]
            n_assays = np.random.choice([1, 2], p=[0.8, 0.2])  # Most compounds have 1 assay
            
            tested_assays = np.random.choice(assay_types, n_assays, replace=False, p=assay_probs)
            
            for assay_type in tested_assays:
                # Generate realistic pIC50/pEC50 values
                if 'IC50' in assay_type:
                    # pIC50 typically 4-9 for drugs
                    base_activity = np.random.normal(6.5, 1.2)
                elif assay_type == 'EC50':
                    # pEC50 similar range but slightly lower
                    base_activity = np.random.normal(6.2, 1.0)
                else:
                    base_activity = np.random.normal(6.0, 1.0)
                
                # Target-specific adjustments
                if target in ['EGFR', 'BRAF']:  # Well-studied targets
                    base_activity += np.random.normal(0.3, 0.5)
                elif target in ['PARP1', 'PLK1']:  # Moderate druggability
                    base_activity += np.random.normal(0.0, 0.3)
                
                # Clamp to reasonable range
                label = np.clip(base_activity, 3.0, 10.0)
                
                # Assign train/val/test splits
                split_probs = [0.8, 0.1, 0.1]  # 80/10/10 split
                split = np.random.choice(['train', 'val', 'test'], p=split_probs)
                
                training_data.append({
                    'ligand_id': compound['ligand_id'],
                    'smiles': compound['smiles'],
                    'target_id': target,
                    'assay_type': assay_type,
                    'label': float(label),
                    'split': split
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    logger.info(f"Generated training dataset:")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Unique compounds: {df['ligand_id'].nunique()}")
    logger.info(f"  Targets: {df['target_id'].nunique()}")
    logger.info(f"  Train/Val/Test: {df['split'].value_counts().to_dict()}")
    
    # Save to file
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved training data to {save_path}")
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate mock training data
    training_data = generate_mock_training_data(
        n_compounds=500,
        n_targets=8
    )
    
    print(f"Generated training data with {len(training_data)} records")
    print(f"Columns: {list(training_data.columns)}")
    print(f"Sample records:")
    print(training_data.head())