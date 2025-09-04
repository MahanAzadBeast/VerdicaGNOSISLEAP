#!/usr/bin/env python3
"""
ChemBERTA Clinical Outcome Predictor
Uses real ChEMBL datasets to train neural networks for predicting clinical outcomes
from SMILES molecular structures
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ChemBERTAClinicalPredictor:
    """ChemBERTA-based clinical outcome predictor using real pharmaceutical data"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        self.model_name = model_name
        self.tokenizer = None
        self.chemberta_model = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🧬 Initializing ChemBERTA Clinical Predictor")
        logger.info(f"📱 Device: {self.device}")
        
    def load_real_pharmaceutical_data(self):
        """Load real ChEMBL pharmaceutical data for training"""
        logger.info("📂 LOADING REAL PHARMACEUTICAL DATA")
        logger.info("=" * 60)
        
        # Load ChEMBL compounds
        chembl_file = "clinical_trial_dataset/data/github_final/chembl_complete_dataset.csv"
        
        try:
            df = pd.read_csv(chembl_file)
            logger.info(f"✅ Loaded ChEMBL: {len(df):,} real pharmaceutical compounds")
            
            # Load clinical trials with SMILES
            trials_file = "clinical_trial_dataset/data/github_final/clinical_trials_with_smiles_complete.csv"
            
            if Path(trials_file).exists():
                trials_df = pd.read_csv(trials_file, low_memory=False)
                trials_with_smiles = trials_df[trials_df['smiles'].notna()].copy()
                logger.info(f"✅ Loaded Clinical Trials: {len(trials_with_smiles):,} trials with SMILES")
                
                # Combine datasets
                combined_df = self._combine_datasets(df, trials_with_smiles)
                logger.info(f"🔗 Combined dataset: {len(combined_df):,} compounds with clinical outcomes")
                
                return combined_df
            else:
                logger.warning("⚠️ Clinical trials with SMILES not found, using ChEMBL only")
                return self._prepare_chembl_only(df)
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def _combine_datasets(self, chembl_df, trials_df):
        """Combine ChEMBL and clinical trials data"""
        logger.info("🔗 Combining ChEMBL and clinical trials data...")
        
        combined_records = []
        
        # Process ChEMBL compounds
        for _, row in chembl_df.iterrows():
            record = {
                'compound_id': row['compound_id'],
                'primary_drug': row['primary_drug'],
                'smiles': row['smiles'],
                'smiles_source': row['smiles_source'],
                
                # Clinical outcome from ChEMBL
                'clinical_phase': row.get('max_clinical_phase'),
                'clinical_outcome': self._determine_clinical_outcome(row.get('max_clinical_phase')),
                'approval_status': 'APPROVED' if row.get('max_clinical_phase') == 4 else 'NOT_APPROVED',
                
                # Molecular properties
                'molecular_weight': row.get('mol_molecular_weight'),
                'logp': row.get('mol_logp'),
                'hbd': row.get('mol_num_hbd'),
                'hba': row.get('mol_num_hba'),
                
                # Data source
                'data_source': 'chembl',
                'has_clinical_outcome': row.get('max_clinical_phase') is not None
            }
            combined_records.append(record)
        
        # Process clinical trials
        for _, row in trials_df.iterrows():
            if pd.notna(row['smiles']):  # Only trials with SMILES
                record = {
                    'compound_id': f"TRIAL_{row['nct_id']}",
                    'primary_drug': row['primary_drug'],
                    'smiles': row['smiles'],
                    'smiles_source': row.get('smiles_source', 'unknown'),
                    
                    # Clinical outcome from trial
                    'clinical_phase': self._parse_trial_phase(row.get('primary_phase')),
                    'clinical_outcome': self._determine_trial_outcome(row),
                    'approval_status': self._determine_approval_from_trial(row),
                    
                    # Trial-specific data
                    'nct_id': row['nct_id'],
                    'trial_status': row.get('overall_status'),
                    'trial_condition': row.get('primary_condition'),
                    
                    # Molecular properties (if available)
                    'molecular_weight': row.get('molecular_weight'),
                    'logp': row.get('logp'),
                    
                    # Data source
                    'data_source': 'clinical_trial',
                    'has_clinical_outcome': True
                }
                combined_records.append(record)
        
        combined_df = pd.DataFrame(combined_records)
        
        # Remove duplicates by SMILES (prefer clinical trial data)
        combined_df = combined_df.sort_values('data_source').drop_duplicates(subset=['smiles'], keep='first')
        
        logger.info(f"🔗 Final combined dataset: {len(combined_df):,} unique compounds")
        
        return combined_df
    
    def _determine_clinical_outcome(self, phase):
        """Determine clinical outcome from phase"""
        if pd.isna(phase):
            return 'UNKNOWN'
        
        phase = float(phase)
        if phase == 4:
            return 'SUCCESS_APPROVED'
        elif phase == 3:
            return 'SUCCESS_LATE_STAGE'
        elif phase in [1, 2]:
            return 'SUCCESS_EARLY_STAGE'
        else:
            return 'UNKNOWN'
    
    def _determine_trial_outcome(self, trial_row):
        """Determine outcome from trial status"""
        status = trial_row.get('overall_status', '').upper()
        
        if status == 'COMPLETED':
            return 'SUCCESS_COMPLETED'
        elif status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
            return 'FAILURE'
        elif status in ['RECRUITING', 'ACTIVE_NOT_RECRUITING']:
            return 'ONGOING'
        else:
            return 'UNKNOWN'
    
    def _parse_trial_phase(self, phase_str):
        """Parse trial phase from string"""
        if pd.isna(phase_str):
            return None
        
        phase_str = str(phase_str).upper()
        if 'PHASE4' in phase_str:
            return 4
        elif 'PHASE3' in phase_str:
            return 3
        elif 'PHASE2' in phase_str:
            return 2
        elif 'PHASE1' in phase_str:
            return 1
        else:
            return None
    
    def _determine_approval_from_trial(self, trial_row):
        """Determine approval status from trial"""
        phase = self._parse_trial_phase(trial_row.get('primary_phase'))
        status = trial_row.get('overall_status', '')
        
        if phase == 4 and status == 'COMPLETED':
            return 'APPROVED'
        elif status in ['TERMINATED', 'WITHDRAWN']:
            return 'FAILED'
        else:
            return 'NOT_APPROVED'
    
    def _prepare_chembl_only(self, chembl_df):
        """Prepare ChEMBL-only dataset if trials not available"""
        logger.info("📊 Preparing ChEMBL-only dataset...")
        
        # Filter for compounds with clinical phase information
        with_phases = chembl_df[chembl_df['max_clinical_phase'].notna()].copy()
        
        with_phases['clinical_outcome'] = with_phases['max_clinical_phase'].apply(self._determine_clinical_outcome)
        with_phases['approval_status'] = with_phases['max_clinical_phase'].apply(
            lambda x: 'APPROVED' if x == 4 else 'NOT_APPROVED'
        )
        
        logger.info(f"📊 ChEMBL compounds with clinical phases: {len(with_phases):,}")
        
        return with_phases
    
    def prepare_training_data(self, df):
        """Prepare data for ChemBERTA training"""
        logger.info("🎯 PREPARING TRAINING DATA FOR CHEMBERTA")
        logger.info("=" * 60)
        
        # Filter for valid SMILES and outcomes
        valid_data = df[
            (df['smiles'].notna()) & 
            (df['smiles'].str.len() > 5) &
            (df['clinical_outcome'] != 'UNKNOWN')
        ].copy()
        
        logger.info(f"✅ Valid training samples: {len(valid_data):,}")
        
        # Prepare features and targets
        X = valid_data['smiles'].tolist()
        y_outcome = valid_data['clinical_outcome'].tolist()
        y_approval = valid_data['approval_status'].tolist()
        
        # Encode labels
        y_outcome_encoded = self.label_encoder.fit_transform(y_outcome)
        
        logger.info(f"🎯 Training targets prepared:")
        logger.info(f"   Clinical outcomes: {len(set(y_outcome))} classes")
        logger.info(f"   Approval status: {len(set(y_approval))} classes")
        
        # Show class distribution
        outcome_counts = pd.Series(y_outcome).value_counts()
        logger.info(f"📊 Clinical outcome distribution:")
        for outcome, count in outcome_counts.items():
            logger.info(f"   {outcome}: {count:,} compounds")
        
        return X, y_outcome_encoded, y_approval, valid_data
    
    def create_chemberta_model(self, num_classes):
        """Create ChemBERTA-based clinical outcome predictor"""
        logger.info("🧠 CREATING CHEMBERTA CLINICAL PREDICTOR")
        
        try:
            # Load ChemBERTA tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.chemberta_model = AutoModel.from_pretrained(self.model_name)
            
            # Freeze ChemBERTA parameters (optional)
            for param in self.chemberta_model.parameters():
                param.requires_grad = False
            
            # Create classification head
            hidden_size = self.chemberta_model.config.hidden_size
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # Move to device
            self.chemberta_model.to(self.device)
            self.classifier.to(self.device)
            
            logger.info(f"✅ ChemBERTA model created with {num_classes} output classes")
            logger.info(f"🔧 Hidden size: {hidden_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating model: {e}")
            return False
    
    def train_clinical_outcome_predictor(self, X, y, epochs=10, batch_size=16):
        """Train ChemBERTA for clinical outcome prediction"""
        logger.info("🚀 TRAINING CHEMBERTA CLINICAL OUTCOME PREDICTOR")
        logger.info("=" * 70)
        
        # Create dataset
        dataset = SMILESDataset(X, y, self.tokenizer)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"📊 Training setup:")
        logger.info(f"   Train samples: {train_size:,}")
        logger.info(f"   Validation samples: {val_size:,}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Epochs: {epochs}")
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.chemberta_model.train()
            self.classifier.train()
            
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (smiles_tokens, labels) in enumerate(train_loader):
                smiles_tokens = {k: v.to(self.device) for k, v in smiles_tokens.items()}
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through ChemBERTA
                with torch.no_grad():
                    chemberta_outputs = self.chemberta_model(**smiles_tokens)
                    molecular_embeddings = chemberta_outputs.last_hidden_state[:, 0, :]  # CLS token
                
                # Forward pass through classifier
                predictions = self.classifier(molecular_embeddings)
                loss = criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    logger.info(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}: Loss {loss.item():.4f}")
            
            # Validation phase
            val_accuracy = self._validate_model(val_loader)
            
            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            logger.info(f"✅ Epoch {epoch+1}/{epochs}:")
            logger.info(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            logger.info(f"   Val Acc: {val_accuracy:.2f}%")
        
        logger.info("🎉 Training complete!")
        return True
    
    def _validate_model(self, val_loader):
        """Validate model performance"""
        self.chemberta_model.eval()
        self.classifier.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for smiles_tokens, labels in val_loader:
                smiles_tokens = {k: v.to(self.device) for k, v in smiles_tokens.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                chemberta_outputs = self.chemberta_model(**smiles_tokens)
                molecular_embeddings = chemberta_outputs.last_hidden_state[:, 0, :]
                predictions = self.classifier(molecular_embeddings)
                
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def predict_clinical_outcome(self, smiles_list):
        """Predict clinical outcomes for unknown SMILES"""
        logger.info(f"🔮 PREDICTING CLINICAL OUTCOMES FOR {len(smiles_list)} SMILES")
        
        self.chemberta_model.eval()
        self.classifier.eval()
        
        predictions = []
        
        with torch.no_grad():
            for smiles in smiles_list:
                # Tokenize SMILES
                tokens = self.tokenizer(
                    smiles,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                # Get molecular embedding
                chemberta_outputs = self.chemberta_model(**tokens)
                molecular_embedding = chemberta_outputs.last_hidden_state[:, 0, :]
                
                # Predict clinical outcome
                outcome_logits = self.classifier(molecular_embedding)
                outcome_probs = torch.softmax(outcome_logits, dim=1)
                
                # Get prediction
                _, predicted_class = torch.max(outcome_logits, 1)
                predicted_outcome = self.label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
                confidence = outcome_probs.max().item()
                
                predictions.append({
                    'smiles': smiles,
                    'predicted_outcome': predicted_outcome,
                    'confidence': confidence,
                    'outcome_probabilities': outcome_probs.cpu().numpy()[0].tolist()
                })
        
        return predictions
    
    def save_trained_model(self, output_dir="clinical_trial_dataset/models"):
        """Save the trained clinical outcome predictor"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        torch.save(self.classifier.state_dict(), output_path / "clinical_outcome_classifier.pth")
        
        # Save label encoder
        import pickle
        with open(output_path / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save model config
        config = {
            "model_name": self.model_name,
            "num_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
            "training_date": datetime.now().isoformat(),
            "data_sources": ["ChEMBL_real_compounds", "Clinical_trials_with_SMILES"],
            "model_type": "ChemBERTA_clinical_outcome_predictor"
        }
        
        with open(output_path / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Model saved to: {output_path}")
        
        return output_path

class SMILESDataset(Dataset):
    """Dataset class for SMILES tokenization"""
    
    def __init__(self, smiles_list, labels, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        
        # Tokenize SMILES
        tokens = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        
        return tokens, torch.tensor(label, dtype=torch.long)

def demonstrate_clinical_prediction():
    """Demonstrate clinical outcome prediction with real data"""
    logger.info("🌟 CHEMBERTA CLINICAL OUTCOME PREDICTION DEMO")
    logger.info("🧬 Using real pharmaceutical data for neural network training")
    logger.info("=" * 80)
    
    # Create predictor
    predictor = ChemBERTAClinicalPredictor()
    
    # Load real pharmaceutical data
    df = predictor.load_real_pharmaceutical_data()
    
    if df.empty:
        logger.error("❌ No data available for training")
        return
    
    # Prepare training data
    X, y_encoded, y_approval, valid_data = predictor.prepare_training_data(df)
    
    # Create model
    num_classes = len(set(y_encoded))
    success = predictor.create_chemberta_model(num_classes)
    
    if not success:
        logger.error("❌ Failed to create model")
        return
    
    # Train model (small demo)
    logger.info("🚀 Starting training demo (2 epochs)...")
    predictor.train_clinical_outcome_predictor(X[:100], y_encoded[:100], epochs=2, batch_size=8)
    
    # Demo prediction on unknown SMILES
    logger.info("🔮 DEMO: Predicting outcomes for unknown SMILES...")
    
    # Use some real SMILES not in training
    demo_smiles = [
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",  # Chloroquine
        "S=C(N/N=C(C1=NC=CC=C1)\\C2=NC=CC=C2)N(C3CCCCC3)C"  # DpC
    ]
    
    predictions = predictor.predict_clinical_outcome(demo_smiles)
    
    for pred in predictions:
        logger.info(f"🧬 SMILES: {pred['smiles'][:40]}...")
        logger.info(f"   Predicted outcome: {pred['predicted_outcome']}")
        logger.info(f"   Confidence: {pred['confidence']:.3f}")
    
    # Save model
    model_path = predictor.save_trained_model()
    logger.info(f"💾 Model saved for future predictions")
    
    return predictor

if __name__ == "__main__":
    demonstrate_clinical_prediction()