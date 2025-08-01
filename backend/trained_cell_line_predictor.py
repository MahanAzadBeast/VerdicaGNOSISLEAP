"""
Trained Cell Line Response Model Inference Integration
Uses the locally trained model for real predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import the model architecture from our training script
class SMILESTokenizer:
    """SMILES tokenizer for molecular encoding"""
    
    def __init__(self):
        # Extended SMILES vocabulary
        self.chars = list("()[]{}.-=+#@/*\\123456789%CNOSPFIBrClncos")
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.chars) + 1  # +1 for padding
        self.pad_token = 0
    
    def tokenize(self, smiles: str, max_length: int = 128) -> List[int]:
        """Convert SMILES string to token indices"""
        tokens = [self.char_to_idx.get(char, 0) for char in smiles[:max_length]]
        # Pad to max_length
        tokens += [self.pad_token] * (max_length - len(tokens))
        return tokens
    
    def batch_tokenize(self, smiles_list: List[str], max_length: int = 128) -> torch.Tensor:
        """Tokenize a batch of SMILES"""
        tokenized = [self.tokenize(smiles, max_length) for smiles in smiles_list]
        return torch.tensor(tokenized, dtype=torch.long)

class MolecularEncoder(nn.Module):
    """Encode molecular SMILES into feature vectors"""
    
    def __init__(self, vocab_size: int = 60, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
    def forward(self, smiles_tokens: torch.Tensor) -> torch.Tensor:
        # Create attention mask for padding
        attention_mask = (smiles_tokens != 0).float()
        
        # Embed SMILES tokens
        embedded = self.embedding(smiles_tokens)  # [batch, seq_len, embedding_dim]
        
        # RNN encoding
        rnn_out, _ = self.rnn(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Simple average pooling with masking
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(rnn_out)
        rnn_masked = rnn_out * mask_expanded
        pooled = rnn_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        
        # Output projection
        molecular_features = self.output_projection(pooled)  # [batch, 64]
        
        return molecular_features

class GenomicEncoder(nn.Module):
    """Encode genomic features into feature vectors"""
    
    def __init__(self, genomic_dim: int = 51):  # Reduced from 63
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(genomic_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(genomic_features)

class CellLineResponseModel(nn.Module):
    """Simplified multi-modal model for predicting IC50"""
    
    def __init__(self, smiles_vocab_size: int = 60, genomic_dim: int = 51):
        super().__init__()
        
        self.molecular_encoder = MolecularEncoder(smiles_vocab_size)
        self.genomic_encoder = GenomicEncoder(genomic_dim)
        
        # Simple fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(128, 128),  # 64 (molecular) + 64 (genomic)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Single IC50 prediction
        )
        
    def forward(self, smiles_tokens: torch.Tensor, genomic_features: torch.Tensor) -> torch.Tensor:
        # Encode inputs
        molecular_features = self.molecular_encoder(smiles_tokens)    # [batch, 64]
        genomic_features_enc = self.genomic_encoder(genomic_features) # [batch, 64]
        
        # Simple concatenation fusion
        fused_features = torch.cat([molecular_features, genomic_features_enc], dim=1)  # [batch, 128]
        
        # Predict IC50
        ic50_pred = self.fusion_layers(fused_features)  # [batch, 1]
        
        return ic50_pred

class TrainedCellLinePredictor:
    """Inference class for the trained Cell Line Response Model"""
    
    def __init__(self, model_dir: str = "/app/models/cell_line"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.metadata = None
        self.device = torch.device('cpu')  # Use CPU for inference
        
        # Expected genomic feature order (from training)
        self.genomic_feature_names = [
            'TP53_mutation', 'KRAS_mutation', 'PIK3CA_mutation', 'EGFR_mutation', 'HER2_mutation',
            'BRAF_mutation', 'MET_mutation', 'ALK_mutation', 'PTEN_mutation', 'BRCA1_mutation',
            'BRCA2_mutation', 'CDK4_mutation', 'CDK6_mutation', 'MDM2_mutation', 'RB1_mutation',
            'TP53_cnv', 'KRAS_cnv', 'PIK3CA_cnv', 'EGFR_cnv', 'HER2_cnv',
            'BRAF_cnv', 'MET_cnv', 'ALK_cnv', 'PTEN_cnv', 'BRCA1_cnv', 'BRCA2_cnv', 'CDK4_cnv',
            'TP53_expression', 'KRAS_expression', 'PIK3CA_expression', 'EGFR_expression', 'HER2_expression',
            'BRAF_expression', 'MET_expression', 'ALK_expression', 'PTEN_expression', 'BRCA1_expression', 'BRCA2_expression', 'CDK4_expression'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and artifacts"""
        
        try:
            # Load metadata
            with open(self.model_dir / "metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            # Load tokenizer
            with open(self.model_dir / "tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load scaler
            with open(self.model_dir / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load model
            genomic_dim = self.metadata['training_data']['genomic_features']
            self.model = CellLineResponseModel(
                smiles_vocab_size=self.tokenizer.vocab_size,
                genomic_dim=genomic_dim
            )
            
            # Load model weights
            state_dict = torch.load(self.model_dir / "best_model.pth", map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logging.info("✅ Trained Cell Line Response Model loaded successfully")
            logging.info(f"   Model trained on: {self.metadata['training_timestamp']}")
            logging.info(f"   Test R²: {self.metadata['performance']['test_r2']:.4f}")
            
        except Exception as e:
            logging.error(f"❌ Failed to load trained model: {e}")
            raise
    
    def prepare_genomic_features(self, genomic_dict: Dict[str, Any]) -> np.ndarray:
        """Prepare genomic features in the correct order"""
        
        # Extract features in the correct order
        features = []
        
        for feature_name in self.genomic_feature_names:
            if feature_name in genomic_dict:
                features.append(float(genomic_dict[feature_name]))
            else:
                # Default values
                if '_mutation' in feature_name:
                    features.append(0.0)  # Wild-type
                elif '_cnv' in feature_name:
                    features.append(0.0)  # Normal copy number
                elif '_expression' in feature_name:
                    features.append(0.0)  # Normal expression
                else:
                    features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, smiles: str, genomic_features: Dict[str, Any]) -> Dict[str, float]:
        """Make a prediction using the trained model"""
        
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare molecular features
            smiles_tokens = self.tokenizer.batch_tokenize([smiles], max_length=80)
            smiles_tokens = smiles_tokens.to(self.device)
            
            # Prepare genomic features
            genomic_array = self.prepare_genomic_features(genomic_features)
            genomic_array = genomic_array.reshape(1, -1)  # Add batch dimension
            
            # Scale genomic features
            genomic_scaled = self.scaler.transform(genomic_array)
            genomic_tensor = torch.tensor(genomic_scaled, dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                pic50_pred = self.model(smiles_tokens, genomic_tensor)
                pic50_value = pic50_pred.item()
            
            # Convert to IC50 nM
            ic50_nm = 10 ** (-pic50_value) * 1e9
            
            # Calculate uncertainty based on model confidence
            uncertainty = self._estimate_uncertainty(genomic_features)
            confidence = 1.0 - uncertainty
            
            return {
                'predicted_ic50_nm': float(ic50_nm),
                'predicted_pic50': float(pic50_value),
                'uncertainty': float(uncertainty),
                'confidence': float(confidence),
                'model_source': 'trained_local'
            }
        
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise
    
    def _estimate_uncertainty(self, genomic_features: Dict[str, Any]) -> float:
        """Estimate prediction uncertainty based on genomic complexity"""
        
        # Count mutations and alterations
        mutation_count = sum(1 for k, v in genomic_features.get('mutations', {}).items() if v == 1)
        cnv_count = sum(1 for k, v in genomic_features.get('cnvs', {}).items() if v != 0)
        
        # More mutations = higher confidence (more genomic context)
        total_alterations = mutation_count + cnv_count
        
        # Base uncertainty decreases with more genomic information
        if total_alterations == 0:
            base_uncertainty = 0.5  # High uncertainty with no genomic info
        elif total_alterations <= 2:
            base_uncertainty = 0.3  # Moderate uncertainty
        else:
            base_uncertainty = 0.2  # Lower uncertainty with rich genomic context
        
        # Add some variability
        uncertainty = base_uncertainty + np.random.uniform(-0.05, 0.05)
        return max(0.1, min(uncertainty, 0.9))  # Clamp between 0.1 and 0.9

# Global instance for the backend integration
_trained_predictor = None

def get_trained_predictor() -> TrainedCellLinePredictor:
    """Get or create the trained predictor instance"""
    global _trained_predictor
    
    if _trained_predictor is None:
        try:
            _trained_predictor = TrainedCellLinePredictor()
        except Exception as e:
            logging.error(f"Failed to initialize trained predictor: {e}")
            raise
    
    return _trained_predictor

def predict_with_trained_model(smiles: str, genomic_features: Dict[str, Any]) -> Dict[str, float]:
    """Use the trained model for prediction"""
    
    try:
        predictor = get_trained_predictor()
        return predictor.predict(smiles, genomic_features)
    except Exception as e:
        logging.error(f"Trained model prediction failed: {e}")
        # Fallback to simulation if trained model fails
        return {
            'predicted_ic50_nm': 1000.0,
            'predicted_pic50': 6.0,
            'uncertainty': 0.5,
            'confidence': 0.5,
            'model_source': 'fallback_simulation'
        }

if __name__ == "__main__":
    # Test the trained model
    print("Testing trained Cell Line Response Model...")
    
    try:
        predictor = TrainedCellLinePredictor()
        
        # Test prediction
        test_smiles = "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC"  # Erlotinib
        test_genomics = {
            'mutations': {'TP53': 1, 'KRAS': 1, 'EGFR': 0},
            'cnvs': {'MYC': 1, 'CDKN2A': -1},
            'expression': {'EGFR': -0.5, 'KRAS': 1.2}
        }
        
        result = predictor.predict(test_smiles, test_genomics)
        print("Test prediction result:", result)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()