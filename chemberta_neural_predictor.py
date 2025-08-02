"""
ChemBERTa Neural Network Predictor
Load and use the trained ChemBERTa Cell Line Response Model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import model architecture (same as training script)
try:
    from transformers import AutoTokenizer, AutoModel
    CHEMBERTA_AVAILABLE = True
except ImportError:
    CHEMBERTA_AVAILABLE = False

class SMILESTokenizer:
    """SMILES tokenizer for drug molecular representation"""
    
    def __init__(self):
        self.chars = list("()[]{}.-=+#@/*\\123456789%CNOSPFIBrClncos")
        self.chars.extend(['A', 'B', 'G', 'H', 'K', 'L', 'M', 'R', 'T', 'V', 'W', 'Y', 'Z'])
        self.chars = list(set(self.chars))
        self.chars.sort()
        
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.chars) + 1
        self.pad_token = 0
    
    def tokenize(self, smiles: str, max_length: int = 128) -> List[int]:
        tokens = [self.char_to_idx.get(char, 0) for char in smiles[:max_length]]
        tokens += [self.pad_token] * (max_length - len(tokens))
        return tokens
    
    def batch_tokenize(self, smiles_list: List[str], max_length: int = 128) -> torch.Tensor:
        tokenized = [self.tokenize(smiles, max_length) for smiles in smiles_list]
        return torch.tensor(tokenized, dtype=torch.long)

class ChemBERTaDrugEncoder(nn.Module):
    """Drug encoder using ChemBERTa or enhanced SMILES encoding"""
    
    def __init__(self, vocab_size: int = 70, use_chemberta: bool = CHEMBERTA_AVAILABLE):
        super().__init__()
        
        self.use_chemberta = use_chemberta and CHEMBERTA_AVAILABLE
        
        if self.use_chemberta:
            try:
                self.chemberta_model_name = "seyonec/ChemBERTa-zinc-base-v1"
                self.tokenizer = AutoTokenizer.from_pretrained(self.chemberta_model_name)
                self.chemberta = AutoModel.from_pretrained(self.chemberta_model_name)
                
                for param in self.chemberta.parameters():
                    param.requires_grad = True
                
                self.chemberta_hidden_size = self.chemberta.config.hidden_size
                
                self.projection = nn.Sequential(
                    nn.Linear(self.chemberta_hidden_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128)
                )
                
            except Exception as e:
                self.use_chemberta = False
        
        if not self.use_chemberta:
            self.embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
            self.rnn = nn.LSTM(256, 512, batch_first=True, bidirectional=True, num_layers=3, dropout=0.3)
            self.attention = nn.MultiheadAttention(1024, num_heads=16, batch_first=True, dropout=0.2)
            self.projection = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128)
            )
    
    def forward(self, smiles_input) -> torch.Tensor:
        if self.use_chemberta:
            if isinstance(smiles_input, list):
                encoded = self.tokenizer(
                    smiles_input,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                device = next(self.parameters()).device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                drug_embedding = self.projection(cls_embedding)
            else:
                raise ValueError("ChemBERTa expects list of SMILES strings")
        else:
            if isinstance(smiles_input, list):
                raise ValueError("Enhanced SMILES encoder expects tokenized tensor")
            
            smiles_tokens = smiles_input
            attention_mask = (smiles_tokens != 0).float()
            
            embedded = self.embedding(smiles_tokens)
            rnn_out, _ = self.rnn(embedded)
            
            attended, _ = self.attention(rnn_out, rnn_out, rnn_out, key_padding_mask=(attention_mask == 0))
            
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(attended)
            attended_masked = attended * mask_expanded
            pooled = attended_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
            
            drug_embedding = self.projection(pooled)
        
        return drug_embedding

class GenomicEncoder(nn.Module):
    """Enhanced genomic encoder for cell line features"""
    
    def __init__(self, genomic_dim: int = 51):
        super().__init__()
        
        mutation_dim = 15
        cnv_dim = 12  
        expr_dim = 12
        meta_dim = genomic_dim - mutation_dim - cnv_dim - expr_dim
        
        self.mutation_encoder = nn.Sequential(
            nn.Linear(mutation_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.cnv_encoder = nn.Sequential(
            nn.Linear(cnv_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.expression_encoder = nn.Sequential(
            nn.Linear(expr_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        if meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            fusion_dim = 192 + 32
        else:
            fusion_dim = 192
        
        self.genomic_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
        self.meta_dim = meta_dim
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        mutations = genomic_features[:, :15]
        cnvs = genomic_features[:, 15:27]
        expression = genomic_features[:, 27:39]
        
        mut_encoded = self.mutation_encoder(mutations)
        cnv_encoded = self.cnv_encoder(cnvs)
        expr_encoded = self.expression_encoder(expression)
        
        genomic_parts = [mut_encoded, cnv_encoded, expr_encoded]
        
        if self.meta_dim > 0:
            meta_features = genomic_features[:, 39:]
            meta_encoded = self.meta_encoder(meta_features)
            genomic_parts.append(meta_encoded)
        
        combined = torch.cat(genomic_parts, dim=1)
        genomic_embedding = self.genomic_fusion(combined)
        
        return genomic_embedding

class ChemBERTaCellLineModel(nn.Module):
    """ChemBERTa-based Cell Line Response Model"""
    
    def __init__(self, smiles_vocab_size: int = 70, genomic_dim: int = 51, use_chemberta: bool = CHEMBERTA_AVAILABLE):
        super().__init__()
        
        self.use_chemberta = use_chemberta
        
        self.drug_encoder = ChemBERTaDrugEncoder(smiles_vocab_size, use_chemberta)
        self.genomic_encoder = GenomicEncoder(genomic_dim)
        
        self.cross_attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True, dropout=0.2)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
    def forward(self, smiles_input, genomic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        drug_embedding = self.drug_encoder(smiles_input)
        genomic_embedding = self.genomic_encoder(genomic_features)
        
        drug_expanded = drug_embedding.unsqueeze(1)
        genomic_expanded = genomic_embedding.unsqueeze(1)
        
        attended_drug, attention_weights = self.cross_attention(
            drug_expanded, genomic_expanded, genomic_expanded
        )
        attended_drug = attended_drug.squeeze(1)
        
        fused_features = torch.cat([attended_drug, genomic_embedding], dim=1)
        
        ic50_pred = self.prediction_head(fused_features)
        uncertainty = self.uncertainty_head(fused_features)
        
        return ic50_pred, uncertainty

class ChemBERTaNeuralPredictor:
    """Trained ChemBERTa Neural Network Predictor"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.device = torch.device('cpu')
        self.model_available = False
        self.metadata = None
        
        # Genomic feature names
        self.genomic_feature_names = [
            'TP53_mutation', 'KRAS_mutation', 'PIK3CA_mutation', 'EGFR_mutation', 'HER2_mutation',
            'BRAF_mutation', 'MET_mutation', 'ALK_mutation', 'PTEN_mutation', 'BRCA1_mutation',
            'BRCA2_mutation', 'CDK4_mutation', 'CDK6_mutation', 'MDM2_mutation', 'RB1_mutation',
            'TP53_cnv', 'KRAS_cnv', 'PIK3CA_cnv', 'EGFR_cnv', 'HER2_cnv',
            'BRAF_cnv', 'MET_cnv', 'ALK_cnv', 'PTEN_cnv', 'BRCA1_cnv', 'BRCA2_cnv', 'CDK4_cnv',
            'TP53_expression', 'KRAS_expression', 'PIK3CA_expression', 'EGFR_expression', 'HER2_expression',
            'BRAF_expression', 'MET_expression', 'ALK_expression', 'PTEN_expression', 'BRCA1_expression', 
            'BRCA2_expression', 'CDK4_expression'
        ]
        
        self.load_trained_model()
    
    def load_trained_model(self):
        """Load the trained ChemBERTa neural network"""
        
        try:
            model_dir = Path("/app/models/chemberta_cell_line")
            metadata_path = model_dir / "chemberta_metadata.json"
            model_path = model_dir / "best_chemberta_model.pth"
            
            if not metadata_path.exists() or not model_path.exists():
                print("⚠️ Trained ChemBERTa model not found - will check again later")
                self.model_available = False
                return
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            print(f"🧬 Loading trained ChemBERTa model...")
            print(f"   📊 Performance: R² = {self.metadata['performance']['final_test_r2']:.4f}")
            print(f"   📊 RMSE: {self.metadata['performance']['final_test_rmse']:.4f}")
            
            # Load tokenizer/scaler
            if self.metadata['chemberta_available']:
                print("   🧬 Using ChemBERTa tokenizer")
                # ChemBERTa tokenizer will be initialized in the model
            else:
                tokenizer_path = model_dir / "smiles_tokenizer.pkl"
                if tokenizer_path.exists():
                    with open(tokenizer_path, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                else:
                    self.tokenizer = SMILESTokenizer()
            
            scaler_path = model_dir / "genomic_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Initialize model architecture
            vocab_size = self.tokenizer.vocab_size if self.tokenizer else 70
            genomic_dim = self.metadata['dataset_info']['genomic_features']
            
            self.model = ChemBERTaCellLineModel(
                smiles_vocab_size=vocab_size,
                genomic_dim=genomic_dim,
                use_chemberta=self.metadata['chemberta_available']
            )
            
            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_available = True
            
            performance_r2 = self.metadata['performance']['final_test_r2']
            performance_category = "EXCELLENT" if performance_r2 > 0.8 else "GOOD" if performance_r2 > 0.6 else "MODERATE" if performance_r2 > 0.3 else "POOR"
            
            print(f"✅ Trained ChemBERTa model loaded successfully!")
            print(f"   📊 Performance: {performance_category} (R² = {performance_r2:.4f})")
            print(f"   🧬 Architecture: {self.metadata['architecture']}")
            print(f"   📋 Training samples: {self.metadata['dataset_info']['training_samples']:,}")
            
        except Exception as e:
            print(f"❌ Failed to load trained ChemBERTa model: {e}")
            self.model_available = False
    
    def prepare_genomic_features(self, genomic_dict: Dict[str, Any]) -> np.ndarray:
        """Prepare genomic features array"""
        
        features = []
        
        for feature_name in self.genomic_feature_names:
            if feature_name in genomic_dict:
                features.append(float(genomic_dict[feature_name]))
            else:
                # Enhanced defaults
                if '_mutation' in feature_name:
                    default_value = 0.0
                elif '_cnv' in feature_name:
                    default_value = 0.0
                elif '_expression' in feature_name:
                    default_value = np.random.normal(0, 0.1)
                else:
                    default_value = 0.0
                
                features.append(default_value)
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, smiles: str, genomic_features: Dict[str, Any]) -> Dict[str, float]:
        """Predict using trained ChemBERTa neural network"""
        
        try:
            if not self.model_available:
                # Try to reload model in case training completed
                self.load_trained_model()
                
                if not self.model_available:
                    return {
                        'predicted_ic50_nm': 1000.0,
                        'predicted_pic50': 6.0,
                        'uncertainty': 0.8,
                        'confidence': 0.2,
                        'model_source': 'training_not_complete'
                    }
            
            # Prepare inputs
            if self.metadata['chemberta_available']:
                smiles_input = [smiles]  # Keep as list for ChemBERTa
            else:
                smiles_tokens = self.tokenizer.batch_tokenize([smiles], max_length=128)
                smiles_input = smiles_tokens.to(self.device)
            
            # Prepare genomic features
            genomic_array = self.prepare_genomic_features(genomic_features)
            genomic_array = genomic_array.reshape(1, -1)
            
            if self.scaler is not None:
                genomic_array = self.scaler.transform(genomic_array)
            
            genomic_tensor = torch.tensor(genomic_array, dtype=torch.float32).to(self.device)
            
            # Neural network prediction
            with torch.no_grad():
                pic50_pred, uncertainty = self.model(smiles_input, genomic_tensor)
                pic50_value = pic50_pred.item()
                uncertainty_value = uncertainty.item()
            
            # Convert to IC50 nM
            ic50_nm = 10 ** (-pic50_value) * 1e9
            confidence = 1.0 - min(uncertainty_value, 0.9)
            
            return {
                'predicted_ic50_nm': float(ic50_nm),
                'predicted_pic50': float(pic50_value),
                'uncertainty': float(uncertainty_value),
                'confidence': float(confidence),
                'model_source': f'trained_chemberta_neural_network_r2_{self.metadata["performance"]["final_test_r2"]:.3f}'
            }
            
        except Exception as e:
            logging.error(f"ChemBERTa neural network prediction error: {e}")
            return {
                'predicted_ic50_nm': 1000.0,
                'predicted_pic50': 6.0,
                'uncertainty': 0.5,
                'confidence': 0.5,
                'model_source': 'neural_network_error'
            }

# Global predictor instance
_chemberta_neural_predictor = None

def get_chemberta_neural_predictor() -> ChemBERTaNeuralPredictor:
    """Get or create the ChemBERTa neural predictor"""
    global _chemberta_neural_predictor
    
    if _chemberta_neural_predictor is None:
        try:
            _chemberta_neural_predictor = ChemBERTaNeuralPredictor()
        except Exception as e:
            logging.error(f"Failed to initialize ChemBERTa neural predictor: {e}")
            raise
    
    return _chemberta_neural_predictor

def predict_with_chemberta_neural_network(smiles: str, genomic_features: Dict[str, Any]) -> Dict[str, float]:
    """Predict using trained ChemBERTa neural network"""
    
    try:
        predictor = get_chemberta_neural_predictor()
        return predictor.predict(smiles, genomic_features)
    except Exception as e:
        logging.error(f"ChemBERTa neural prediction failed: {e}")
        return {
            'predicted_ic50_nm': 1000.0,
            'predicted_pic50': 6.0,
            'uncertainty': 0.5,
            'confidence': 0.5,
            'model_source': 'neural_network_fallback'
        }

if __name__ == "__main__":
    # Test the neural predictor
    print("🧬 Testing ChemBERTa Neural Network Predictor...")
    
    try:
        predictor = ChemBERTaNeuralPredictor()
        
        # Test prediction
        test_smiles = "CC(C)[C@H](C(=O)N1CC2=C(C1=O)SC(=N2)C3=CC(=C(C=C3)F)F)N4C(=O)C5=C(C4=O)C=CC=C5I"  # Trametinib
        test_genomics = {
            'TP53_mutation': 1,
            'KRAS_mutation': 1,
            'EGFR_mutation': 0,
            'MYC_cnv': 1,
            'KRAS_expression': 1.5
        }
        
        result = predictor.predict(test_smiles, test_genomics)
        print("✅ ChemBERTa neural prediction result:")
        print(f"   IC50: {result['predicted_ic50_nm']:.1f} nM")
        print(f"   pIC50: {result['predicted_pic50']:.2f}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Source: {result['model_source']}")
        
    except Exception as e:
        print(f"❌ Neural predictor test failed: {e}")
        import traceback
        traceback.print_exc()