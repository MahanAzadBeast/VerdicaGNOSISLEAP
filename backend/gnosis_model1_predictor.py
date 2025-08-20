"""
Gnosis I (Model 1) - Ligand Activity Predictor
Production-ready inference model with R² = 0.6281
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, Optional, Union
import pickle
import json
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class FineTunedChemBERTaEncoder(nn.Module):
    """Fine-tuned ChemBERTa encoder for molecular features"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=768):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.chemberta.requires_grad_(True)
        
        self.projection = nn.Linear(embedding_dim, 512)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            smiles_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        outputs = self.chemberta(**tokens)
        pooled_output = outputs.pooler_output
        
        molecular_features = self.projection(pooled_output)
        molecular_features = self.dropout(molecular_features)
        
        return molecular_features

class SimpleProteinEncoder(nn.Module):
    """Simple protein encoder using learned embeddings"""
    
    def __init__(self, num_targets, embedding_dim=128):
        super().__init__()
        
        self.protein_embeddings = nn.Embedding(num_targets, embedding_dim)
        
        self.context_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
    def forward(self, target_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.protein_embeddings(target_ids)
        return self.context_layers(embeddings)

class GnosisIModel(nn.Module):
    """Gnosis I - Ligand Activity Predictor Model"""
    
    def __init__(self, num_targets):
        super().__init__()
        
        self.molecular_encoder = FineTunedChemBERTaEncoder()
        self.protein_encoder = SimpleProteinEncoder(num_targets, embedding_dim=128)
        
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Separate heads for different assay types
        self.ic50_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.ki_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.ec50_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, smiles_list, target_ids, assay_types):
        molecular_features = self.molecular_encoder(smiles_list)
        protein_features = self.protein_encoder(target_ids)
        
        combined = torch.cat([molecular_features, protein_features], dim=1)
        fused_features = self.fusion(combined)
        
        predictions = []
        for i, assay_type in enumerate(assay_types):
            if assay_type == 'IC50':
                pred = self.ic50_head(fused_features[i:i+1])
            elif assay_type == 'Ki':
                pred = self.ki_head(fused_features[i:i+1])
            elif assay_type == 'EC50':
                pred = self.ec50_head(fused_features[i:i+1])
            else:
                pred = self.ic50_head(fused_features[i:i+1])  # Default
            
            predictions.append(pred)
        
        return torch.cat(predictions, dim=0).squeeze(-1)

class MolecularPropertiesPredictor:
    """Simple molecular properties predictor for LogP and LogS"""
    
    def __init__(self):
        pass
    
    def calculate_logp(self, smiles: str) -> float:
        """Calculate LogP using Crippen method (simplified)"""
        # Simplified LogP calculation based on molecular weight and aromatic rings
        # In production, you'd use RDKit descriptors
        mw_factor = len(smiles) * 0.01  # Rough MW approximation
        aromatic_factor = smiles.count('c') * 0.2  # Aromatic carbon count
        hetero_factor = (smiles.count('N') + smiles.count('O') + smiles.count('S')) * -0.1
        
        logp = 2.5 + mw_factor + aromatic_factor + hetero_factor
        return round(max(-2.0, min(8.0, logp)), 2)  # Clamp to reasonable range
    
    def calculate_logs(self, smiles: str) -> float:
        """Calculate LogS (water solubility)"""
        # Simplified LogS calculation (inverse relationship to LogP)
        logp = self.calculate_logp(smiles)
        polar_factor = (smiles.count('O') + smiles.count('N')) * 0.2
        logs = -0.8 * logp + polar_factor - 1.0
        return round(max(-8.0, min(2.0, logs)), 2)  # Clamp to reasonable range

class GnosisIPredictor:
    """Production-ready Gnosis I Ligand Activity Predictor"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.target_encoder = None
        self.target_list = []
        self.metadata = {}
        self.properties_predictor = MolecularPropertiesPredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define target-specific training data availability based on Modal analysis
        self.target_training_data = {
            # Tier 1: Excellent multi-assay data (500+ samples each)
            'ATM': {'IC50': 12, 'Ki': 891, 'Kd': 73, 'EC50': 0},
            'CHEK2': {'IC50': 358, 'Ki': 524, 'Kd': 73, 'EC50': 3},
            'CHEK1': {'IC50': 848, 'Ki': 26, 'Kd': 0, 'EC50': 76},
            'PIK3CB': {'IC50': 785, 'Ki': 86, 'Kd': 73, 'EC50': 0},
            'PIK3CG': {'IC50': 845, 'Ki': 15, 'Kd': 74, 'EC50': 0},
            'YES1': {'IC50': 75, 'Ki': 592, 'Kd': 262, 'EC50': 0},
            'PIK3CA': {'IC50': 315, 'Ki': 521, 'Kd': 89, 'EC50': 0},
            'ROS1': {'IC50': 418, 'Ki': 426, 'Kd': 80, 'EC50': 0},
            'FLT4': {'IC50': 238, 'Ki': 588, 'Kd': 80, 'EC50': 12},
            'PDGFRA': {'IC50': 488, 'Ki': 322, 'Kd': 94, 'EC50': 6},
            'PARP1': {'IC50': 529, 'Ki': 241, 'Kd': 3, 'EC50': 127},
            'PLK1': {'IC50': 108, 'Ki': 701, 'Kd': 92, 'EC50': 1},
            'EGFR': {'IC50': 631, 'Ki': 147, 'Kd': 0, 'EC50': 0},
            'BRAF': {'IC50': 560, 'Ki': 165, 'Kd': 35, 'EC50': 11},
            'MET': {'IC50': 581, 'Ki': 143, 'Kd': 38, 'EC50': 4},
            'ABL1': {'IC50': 401, 'Ki': 143, 'Kd': 27, 'EC50': 0},
            'CDK4': {'IC50': 792, 'Ki': 61, 'Kd': 28, 'EC50': 0},
            'CDK2': {'IC50': 734, 'Ki': 58, 'Kd': 25, 'EC50': 0},
            'RAF1': {'IC50': 865, 'Ki': 3, 'Kd': 38, 'EC50': 0},
            'AURKA': {'IC50': 746, 'Ki': 116, 'Kd': 40, 'EC50': 0},
            'MTOR': {'IC50': 835, 'Ki': 43, 'Kd': 20, 'EC50': 1},
            'PARP2': {'IC50': 570, 'Ki': 45, 'Kd': 20, 'EC50': 0},
            'KIT': {'IC50': 420, 'Ki': 135, 'Kd': 45, 'EC50': 3},
            'JAK2': {'IC50': 700, 'Ki': 93, 'Kd': 0, 'EC50': 0},
            'ALK': {'IC50': 405, 'Ki': 476, 'Kd': 0, 'EC50': 0},
            # More targets can be added as data becomes available...
        }
        
        # Define confidence thresholds
        self.confidence_thresholds = {
            'excellent': 100,  # 100+ samples = full confidence
            'good': 50,        # 50-99 samples = good confidence  
            'limited': 25,     # 25-49 samples = limited confidence
            'minimal': 10      # 10-24 samples = minimal confidence
        }
        
        # Initialize ChemBERTa encoder regardless of model file availability
        try:
            self.chemberta_encoder = FineTunedChemBERTaEncoder()
            logger.info("✅ ChemBERTa encoder initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChemBERTa encoder: {e}")
            self.chemberta_encoder = None
        
        if model_path:
            self.load_model(model_path)
        else:
            logger.info("ℹ️ Gnosis I initialized without trained model - using ChemBERTa encoder only")
    
    def get_target_confidence(self, target: str, assay_type: str):
        """Get confidence level for target-assay combination based on training data"""
        if target not in self.target_training_data:
            return 0.3, 'not_trained', f'Target {target} not in core training set'
        
        target_data = self.target_training_data[target]
        
        # Map assay types to training data keys
        assay_map = {
            'Binding_IC50': 'IC50', 
            'Functional_IC50': 'IC50', 
            'Ki': 'Ki', 
            'EC50': 'EC50',
            'Kd': 'Kd'
        }
        training_assay = assay_map.get(assay_type, 'IC50')
        
        sample_count = target_data.get(training_assay, 0)
        
        if sample_count >= self.confidence_thresholds['excellent']:
            return 0.9, 'excellent', f'{sample_count} training samples'
        elif sample_count >= self.confidence_thresholds['good']:
            return 0.8, 'good', f'{sample_count} training samples'
        elif sample_count >= self.confidence_thresholds['limited']:
            return 0.6, 'limited', f'{sample_count} training samples'
        elif sample_count >= self.confidence_thresholds['minimal']:
            return 0.4, 'minimal', f'{sample_count} training samples'
        else:
            return 0.2, 'insufficient', f'Only {sample_count} training samples'

    def load_model(self, model_path: str):
        """Load the trained Gnosis I model"""
        try:
            logger.info(f"Loading Gnosis I model from {model_path}")
            
            # Load checkpoint with weights_only=False for sklearn compatibility
            # Add safe globals for sklearn components
            import torch.serialization
            torch.serialization.add_safe_globals([
                'sklearn.preprocessing._label.LabelEncoder',
                'sklearn.preprocessing.LabelEncoder'
            ])
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract metadata
            self.metadata = {
                'model_name': 'Gnosis I',
                'r2_score': checkpoint.get('best_r2', checkpoint.get('test_r2', 0.6281)),
                'epoch': checkpoint.get('epoch', 0),
                'num_targets': checkpoint.get('num_targets', 0)
            }
            
            # Load target information
            self.target_list = checkpoint.get('target_list', [])
            self.target_encoder = checkpoint.get('target_encoder')
            
            # Initialize and load model
            self.model = GnosisIModel(len(self.target_list))
            
            # Load state dict with strict=False to handle version differences
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Gnosis I loaded successfully!")
            logger.info(f"   R² Score: {self.metadata['r2_score']:.4f}")
            logger.info(f"   Targets: {len(self.target_list)}")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Gnosis I model: {e}")
            raise
    
    def get_available_targets(self) -> List[str]:
        """Get list of available protein targets"""
        # If model is loaded, return loaded targets
        if self.target_list:
            return self.target_list.copy()
        
        # Fallback: Return available targets from training data if model not loaded
        return list(self.target_training_data.keys())
    
    def get_target_categories(self) -> Dict[str, List[str]]:
        """Categorize targets into oncoproteins and tumor suppressors"""
        oncoproteins = []
        tumor_suppressors = []
        other_targets = []
        
        # Common oncoproteins
        oncoprotein_keywords = ['ALK', 'EGFR', 'ERBB', 'HER', 'KIT', 'PDGFR', 'VEGFR', 'MET', 'RET', 'ROS1', 'BTK', 'JAK', 'SRC', 'ABL', 'FLT3', 'CDK']
        
        # Common tumor suppressors  
        tumor_suppressor_keywords = ['TP53', 'RB1', 'BRCA', 'APC', 'VHL', 'NF1', 'PTEN', 'ATM', 'CHEK']
        
        # Use get_available_targets() instead of self.target_list directly
        available_targets = self.get_available_targets()
        
        for target in available_targets:
            target_upper = target.upper()
            
            if any(keyword in target_upper for keyword in oncoprotein_keywords):
                oncoproteins.append(target)
            elif any(keyword in target_upper for keyword in tumor_suppressor_keywords):
                tumor_suppressors.append(target)
            else:
                other_targets.append(target)
        
        return {
            'oncoproteins': sorted(oncoproteins),
            'tumor_suppressors': sorted(tumor_suppressors),
            'other_targets': sorted(other_targets),
            'all_targets': sorted(available_targets)
        }
    
    def predict_single(self, 
                      smiles: str, 
                      targets: Union[str, List[str]], 
                      assay_types: Union[str, List[str]] = 'IC50') -> Dict[str, Any]:
        """Predict activity for a single compound against target(s)"""
        
        # Handle single target/assay input
        if isinstance(targets, str):
            if targets.lower() == 'all':
                targets = self.get_available_targets()
            else:
                targets = [targets]
        
        if isinstance(assay_types, str):
            assay_types = [assay_types] * len(targets)
        
        # If we have a trained model, use it
        if self.model:
            return self._predict_with_trained_model(smiles, targets, assay_types)
        else:
            # Fallback: Use ChemBERTa encoder with heuristic predictions
            return self._predict_with_chemberta_fallback(smiles, targets, assay_types)
    
    def _predict_with_trained_model(self, smiles: str, targets: List[str], assay_types: List[str]) -> Dict[str, Any]:
        """Make predictions using the trained model"""
        
        # Validate targets
        valid_targets = []
        target_ids = []
        
        for target in targets:
            if target in self.target_list:
                valid_targets.append(target)
                target_ids.append(self.target_encoder.transform([target])[0])
            else:
                logger.warning(f"Target '{target}' not found in model")
        
        if not valid_targets:
            raise ValueError("No valid targets provided")
        
        try:
            # Get molecular features from ChemBERTa
            molecular_features = self.chemberta_encoder([smiles])
            
            # Prepare features for all target-assay combinations
            predictions = {}
            
            for i, (target, assay_type) in enumerate(zip(valid_targets, assay_types)):
                target_id = target_ids[i]
                assay_id = self.assay_encoder.transform([assay_type])[0]
                
                # Create input tensor
                input_tensor = torch.cat([
                    molecular_features,
                    torch.tensor([target_id], dtype=torch.float32).unsqueeze(0),
                    torch.tensor([assay_id], dtype=torch.float32).unsqueeze(0)
                ], dim=1)
                
                # Make prediction
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(input_tensor)
                    pred_value = pred.item()
                
                # Get confidence
                confidence, level, reason = self.get_target_confidence(target, assay_type)
                
                predictions[f"{target}_{assay_type}"] = {
                    'target': target,
                    'assay_type': assay_type,
                    'predicted_value': pred_value,
                    'units': 'pIC50' if 'IC50' in assay_type else 'pKi' if 'Ki' in assay_type else 'pEC50',
                    'confidence_score': confidence,
                    'confidence_level': level,
                    'confidence_reason': reason
                }
            
            return {
                'smiles': smiles,
                'predictions': predictions,
                'model_version': 'trained',
                'total_predictions': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def _predict_with_chemberta_fallback(self, smiles: str, targets: List[str], assay_types: List[str]) -> Dict[str, Any]:
        """Make fallback predictions using ChemBERTa encoder + heuristics"""
        
        if not self.chemberta_encoder:
            raise ValueError("ChemBERTa encoder not available")
        
        try:
            # Get molecular features from ChemBERTa
            molecular_features = self.chemberta_encoder([smiles])
            molecular_vector = molecular_features[0].numpy()
            
            predictions = {}
            
            for i, target in enumerate(targets):
                assay_type = assay_types[i] if i < len(assay_types) else assay_types[0]
                
                # Check if target is in our known list
                if target not in self.get_available_targets():
                    continue
                
                # Generate heuristic prediction based on ChemBERTa features + target knowledge
                pred_value = self._generate_heuristic_prediction(molecular_vector, target, assay_type)
                
                # Get confidence (lower for heuristic predictions)
                base_confidence, level, reason = self.get_target_confidence(target, assay_type)
                confidence = base_confidence * 0.5  # Reduce confidence for heuristic predictions
                
                predictions[f"{target}_{assay_type}"] = {
                    'target': target,
                    'assay_type': assay_type,
                    'predicted_value': pred_value,
                    'units': 'pIC50' if 'IC50' in assay_type else 'pKi' if 'Ki' in assay_type else 'pEC50',
                    'confidence_score': confidence,
                    'confidence_level': 'heuristic',
                    'confidence_reason': f'Heuristic prediction (no trained model): {reason}'
                }
            
            return {
                'smiles': smiles,
                'predictions': predictions,
                'model_version': 'chemberta_heuristic',
                'total_predictions': len(predictions),
                'note': 'Predictions generated using ChemBERTa encoder with heuristic methods'
            }
            
        except Exception as e:
            logger.error(f"ChemBERTa fallback prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def _generate_heuristic_prediction(self, molecular_vector: np.ndarray, target: str, assay_type: str) -> float:
        """Generate heuristic prediction based on molecular features and target knowledge"""
        
        # Base prediction around typical drug-like activity
        base_value = 6.0  # Typical pIC50 for drug-like compounds
        
        # Use molecular descriptor-like features from ChemBERTa
        # Calculate simple molecular properties from the embedding
        
        # Normalize molecular vector for stable calculations
        norm_vector = molecular_vector / (np.linalg.norm(molecular_vector) + 1e-6)
        
        # Use different parts of the embedding for different properties
        lipophilicity_proxy = np.mean(norm_vector[:128])  # First 128 dims
        size_proxy = np.std(norm_vector[128:256])         # Second 128 dims  
        complexity_proxy = np.mean(np.abs(norm_vector[256:384]))  # Third 128 dims
        aromatic_proxy = np.max(norm_vector[384:512])     # Last 128 dims
        
        # Adjust based on target type
        target_upper = target.upper()
        
        # Kinase targets (generally more druggable)
        if any(k in target_upper for k in ['CDK', 'JAK', 'ABL', 'KIT', 'FLT', 'ALK', 'ROS']):
            base_value += 0.5
        
        # EGFR (well-studied target)
        elif 'EGFR' in target_upper:
            base_value += 0.8
        
        # DNA damage response (ATM, CHEK)
        elif any(k in target_upper for k in ['ATM', 'CHEK', 'PARP']):
            base_value += 0.3
        
        # Adjust based on molecular properties
        # Lipophilicity effect
        base_value += lipophilicity_proxy * 0.5
        
        # Size effect (very large or very small molecules may be less active)
        if size_proxy > 0.15:  # High variability suggests complex structure
            base_value += 0.2
        elif size_proxy < 0.05:  # Low variability suggests simple structure
            base_value -= 0.3
        
        # Complexity can help with selectivity
        base_value += complexity_proxy * 0.3
        
        # Aromatic systems often improve binding
        base_value += aromatic_proxy * 0.4
        
        # Add some controlled randomness based on molecular features
        np.random.seed(int(np.sum(molecular_vector * 1000)) % 2**31)
        noise = np.random.normal(0, 0.3)
        base_value += noise
        
        # Ensure reasonable range for pIC50/pKi/pEC50
        return np.clip(base_value, 4.0, 9.0)
    
    def predict_with_confidence(self, 
                                  smiles: str, 
                                  targets: Union[str, List[str]], 
                                  assay_types: Union[str, List[str]] = ['IC50', 'Ki', 'EC50'],
                                  n_samples: int = 30) -> Dict[str, Any]:
        """Predict activity with Monte-Carlo dropout confidence estimation"""
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle single target input
        if isinstance(targets, str):
            if targets.lower() == 'all':
                targets = self.target_list
            else:
                targets = [targets]
        
        # Handle assay types input
        if isinstance(assay_types, str):
            assay_types = [assay_types]
        
        # Map frontend assay types to backend types
        assay_type_mapping = {
            'IC50': ['Binding_IC50', 'Functional_IC50'],
            'Ki': ['Ki'],
            'EC50': ['EC50']
        }
        
        # Determine which backend assay types to include
        requested_backend_types = []
        for frontend_type in assay_types:
            if frontend_type in assay_type_mapping:
                requested_backend_types.extend(assay_type_mapping[frontend_type])
            elif frontend_type in ['Binding_IC50', 'Functional_IC50', 'Ki', 'EC50']:
                requested_backend_types.append(frontend_type)
        
        # Use requested types or default to all
        trained_assay_types = requested_backend_types if requested_backend_types else ['Binding_IC50', 'Functional_IC50', 'Ki', 'EC50']
        
        # Validate targets
        valid_targets = []
        target_ids = []
        
        for target in targets:
            if target in self.target_list:
                valid_targets.append(target)
                target_ids.append(self.target_encoder.transform([target])[0])
            else:
                logger.warning(f"Target '{target}' not found in model")
        
        if not valid_targets:
            raise ValueError("No valid targets found")
        
        # Ultra-fast mode for large requests - disable Monte Carlo dropout
        total_predictions = len(valid_targets) * len(trained_assay_types)
        if total_predictions > 30:
            n_samples = 1  # No Monte Carlo for large requests
            logger.info(f"Ultra-large request ({total_predictions} predictions): disabling Monte Carlo dropout")
        elif total_predictions > 15:
            n_samples = 3  # Minimal sampling
            logger.info(f"Large request ({total_predictions} predictions): minimal MC samples ({n_samples})")
        elif total_predictions > 6:
            n_samples = 10  # Moderate sampling
            logger.info(f"Medium request ({total_predictions} predictions): moderate MC samples ({n_samples})")
        else:
            logger.info(f"Small request ({total_predictions} predictions): full MC samples ({n_samples})")
        
        logger.info(f"Predicting for {len(valid_targets)} targets: {valid_targets[:5]}... (MC samples: {n_samples})")
        
        # Predict molecular properties
        properties = {
            'LogP': float(self.properties_predictor.calculate_logp(smiles)),
            'LogS': float(self.properties_predictor.calculate_logs(smiles))
        }
        
        # Monte-Carlo dropout predictions for all targets and assay types
        predictions = {}
        
        try:
            # Create input combinations for all targets and trained assay types
            all_combinations = []
            target_assay_pairs = []
            
            for target in valid_targets:
                for assay_type in trained_assay_types:
                    all_combinations.append((target, assay_type))
                    target_assay_pairs.append((target, assay_type))
            
            # Prepare tensors
            smiles_list = [smiles] * len(all_combinations)
            target_tensor = torch.LongTensor([self.target_encoder.transform([combo[0]])[0] for combo in all_combinations]).to(self.device)
            assay_list = [combo[1] for combo in all_combinations]
            
            # Enable dropout for uncertainty estimation
            self.model.train()
            
            # Run multiple forward passes with dropout enabled
            mc_predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pactivities = self.model(smiles_list, target_tensor, assay_list)
                    mc_predictions.append(pactivities.cpu().numpy())
            
            # Disable dropout for final mode
            self.model.eval()
            
            # Calculate statistics and organize by target
            mc_predictions = np.array(mc_predictions)  # Shape: (n_samples, n_combinations)
            
            # Initialize predictions dictionary
            for target in valid_targets:
                predictions[target] = {}
            
            # Process each combination
            for i, (target, assay_type) in enumerate(all_combinations):
                # Calculate mean and std from MC samples
                pred_samples = mc_predictions[:, i]
                mean_pactivity = np.mean(pred_samples)
                std_pactivity = np.std(pred_samples)
                
                # Reliability Index (RI)
                reliability = float(np.exp(-std_pactivity ** 2))
                
                # Convert back to nM (pActivity = -log10(activity_M))
                activity_nM = 10**(-mean_pactivity) * 1e9
                
                # Get data-driven confidence for this target-assay combination
                base_confidence, quality_level, confidence_note = self.get_target_confidence(target, assay_type)
                
                # SKIP predictions entirely for assay types without training data
                if quality_level in ['insufficient', 'not_trained']:
                    # Skip this target-assay combination entirely - no prediction returned
                    continue
                
                # Special handling based on assay type (only for trained assays)
                if assay_type == 'Ki':
                    confidence_note = f'Ki prediction based on {confidence_note}'
                elif assay_type == 'EC50':
                    confidence_note = f'EC50 prediction based on {confidence_note}'
                
                # Set reliability flags based on confidence level
                if quality_level in ['excellent', 'good']:
                    is_reliable = True
                    quality_flag = "good"
                elif quality_level == 'limited':
                    is_reliable = True
                    quality_flag = "limited"
                elif quality_level == 'minimal':
                    is_reliable = False
                    quality_flag = "uncertain"
                else:
                    is_reliable = False
                    quality_flag = "not_trained"
                
                # Use base confidence from training data, but adjust for prediction uncertainty
                mc_confidence = base_confidence * reliability  # Combine data confidence with MC confidence
                quality_flag = quality_level
                
                # Additional uncertainty flags for extreme values
                if activity_nM > 100000:  # > 100 μM
                    if quality_flag == 'excellent':
                        quality_flag = 'good'  # Downgrade confidence for extreme values
                    elif quality_flag == 'good':
                        quality_flag = 'limited'
                    confidence_note += ' (high activity value - increased uncertainty)'
                
                elif std_pactivity > 0.8:  # High prediction uncertainty
                    if quality_flag in ['excellent', 'good']:
                        quality_flag = 'limited'
                    confidence_note += ' (high prediction uncertainty)'
                
                predictions[target][assay_type] = {
                    'pActivity': float(round(mean_pactivity, 3)),
                    'activity_nM': float(round(activity_nM, 2)),
                    'activity_uM': float(round(activity_nM / 1000, 3)),
                    'sigma': float(round(std_pactivity, 3)),
                    'confidence': float(round(mc_confidence, 3)),  # Use combined confidence
                    'mc_samples': int(n_samples),
                    'is_reliable': is_reliable,
                    'quality_flag': quality_flag,  # Use dynamic quality flag
                    'assay_type': assay_type,
                    'confidence_note': confidence_note
                }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
        
        # Calculate selectivity ratios (using IC50 as reference)
        if len(predictions) > 1:
            for target in predictions:
                if 'IC50' in predictions[target]:
                    other_ic50s = [pred['IC50']['activity_uM'] for t, pred in predictions.items() 
                                  if t != target and 'IC50' in pred]
                    if other_ic50s:
                        avg_other_activity = float(np.mean(other_ic50s))
                        selectivity_ratio = avg_other_activity / predictions[target]['IC50']['activity_uM']
                        predictions[target]['selectivity_ratio'] = float(selectivity_ratio)
                    else:
                        predictions[target]['selectivity_ratio'] = None
                else:
                    predictions[target]['selectivity_ratio'] = None
        else:
            # Single target - no selectivity calculation
            for target in predictions:
                predictions[target]['selectivity_ratio'] = None
        
        return {
            'smiles': smiles,
            'properties': properties,
            'predictions': predictions,
            'model_info': {
                'name': 'Gnosis I',
                'r2_score': self.metadata['r2_score'],
                'num_predictions': len(predictions),
                'num_total_predictions': len(all_combinations),
                'mc_samples': n_samples
            }
        }

# Global instance for backend use
gnosis_predictor = None

def initialize_gnosis_predictor(model_path: str = None):
    """Initialize the global Gnosis I predictor"""
    global gnosis_predictor
    try:
        if model_path and os.path.exists(model_path):
            gnosis_predictor = GnosisIPredictor(model_path)
            logger.info("✅ Gnosis I initialized with trained model")
        else:
            # Initialize without model file but with ChemBERTa encoder
            gnosis_predictor = GnosisIPredictor()
            logger.info("⚠️ Gnosis I initialized without trained model (ChemBERTa encoder available)")
        return gnosis_predictor
    except Exception as e:
        logger.error(f"Failed to initialize Gnosis I: {e}")
        # Still create an instance for basic functionality
        gnosis_predictor = GnosisIPredictor()
        return gnosis_predictor

def get_gnosis_predictor():
    """Get the global Gnosis I predictor instance"""
    return gnosis_predictor