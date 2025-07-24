"""
Enhanced Graph Neural Network for Molecular Property Prediction
Custom implementation with molecular graphs and message passing
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from chembl_data_manager import chembl_manager

logger = logging.getLogger(__name__)

class MolecularGraphData:
    """Custom molecular graph representation"""
    
    def __init__(self, atom_features, edge_indices, edge_features):
        self.atom_features = torch.FloatTensor(atom_features)
        self.edge_indices = torch.LongTensor(edge_indices)
        self.edge_features = torch.FloatTensor(edge_features) if edge_features is not None else None
        self.num_atoms = len(atom_features)
    
    def to(self, device):
        """Move graph data to device"""
        self.atom_features = self.atom_features.to(device)
        self.edge_indices = self.edge_indices.to(device)
        if self.edge_features is not None:
            self.edge_features = self.edge_features.to(device)
        return self

class MolecularGraphBuilder:
    """Build molecular graphs from SMILES for enhanced GNN processing"""
    
    # Enhanced atom features
    ATOM_FEATURES = {
        'atomic_num': list(range(1, 119)),  # 1-118 elements
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ],
        'aromaticity': [False, True],
        'num_hs': [0, 1, 2, 3, 4],
        'in_ring': [False, True]
    }
    
    def __init__(self):
        self.atom_feat_dim = sum(len(feat_list) for feat_list in self.ATOM_FEATURES.values())
        logger.info(f"🔬 Enhanced atom feature dimension: {self.atom_feat_dim}")
    
    def _safe_one_hot(self, value, choices):
        """Safe one-hot encoding with fallback for unknown values"""
        try:
            return [float(value == choice) for choice in choices]
        except:
            return [0.0] * len(choices)
    
    def _get_atom_features(self, atom):
        """Extract enhanced atom features"""
        features = []
        
        # Atomic number
        features.extend(self._safe_one_hot(atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num']))
        
        # Degree
        features.extend(self._safe_one_hot(atom.GetDegree(), self.ATOM_FEATURES['degree']))
        
        # Formal charge
        features.extend(self._safe_one_hot(atom.GetFormalCharge(), self.ATOM_FEATURES['formal_charge']))
        
        # Hybridization
        features.extend(self._safe_one_hot(atom.GetHybridization(), self.ATOM_FEATURES['hybridization']))
        
        # Aromaticity
        features.extend(self._safe_one_hot(atom.GetIsAromatic(), self.ATOM_FEATURES['aromaticity']))
        
        # Number of hydrogens
        features.extend(self._safe_one_hot(atom.GetTotalNumHs(), self.ATOM_FEATURES['num_hs']))
        
        # In ring
        features.extend(self._safe_one_hot(atom.IsInRing(), self.ATOM_FEATURES['in_ring']))
        
        return features
    
    def smiles_to_graph(self, smiles: str) -> Optional[MolecularGraphData]:
        """Convert SMILES to molecular graph"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for complete molecular representation
            mol = Chem.AddHs(mol)
            
            # Extract atom features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_feat = self._get_atom_features(atom)
                atom_features.append(atom_feat)
            
            # Extract bonds and create edge list
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                # Add both directions for undirected graph
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                edge_indices.extend([[i, j], [j, i]])
                
                # Simple bond features
                bond_type_onehot = [0, 0, 0]  # single, double, triple
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    bond_type_onehot[0] = 1
                elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    bond_type_onehot[1] = 1
                elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    bond_type_onehot[2] = 1
                
                is_aromatic = float(bond.GetIsAromatic())
                is_conjugated = float(bond.GetIsConjugated())
                
                bond_feat = bond_type_onehot + [is_aromatic, is_conjugated]
                edge_features.extend([bond_feat, bond_feat])  # Same features for both directions
            
            # Handle single-atom molecules
            if len(edge_indices) == 0:
                edge_indices = [[0, 0]]  # Self-loop
                edge_features = [[0, 0, 0, 0, 0]]  # Dummy edge features
            
            # Transpose edge indices for proper format
            edge_indices = np.array(edge_indices).T
            
            return MolecularGraphData(atom_features, edge_indices, edge_features)
            
        except Exception as e:
            logger.warning(f"Error converting SMILES to graph: {smiles} - {e}")
            return None

class GraphConvolutionLayer(nn.Module):
    """Custom graph convolution layer with message passing"""
    
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Message and update networks
        self.message_net = nn.Linear(in_features * 2, out_features)
        self.update_net = nn.Linear(in_features + out_features, out_features)
        
    def forward(self, atom_features, edge_indices):
        """Forward pass with message passing"""
        num_atoms = atom_features.size(0)
        
        # Initialize messages
        messages = torch.zeros(num_atoms, self.out_features, device=atom_features.device)
        
        # Message passing
        for i in range(edge_indices.size(1)):
            src, dst = edge_indices[0, i], edge_indices[1, i]
            
            # Create message from source to destination
            src_feat = atom_features[src]
            dst_feat = atom_features[dst]
            
            # Concatenate source and destination features
            edge_input = torch.cat([src_feat, dst_feat], dim=0)
            message = self.message_net(edge_input)
            
            # Aggregate message at destination
            messages[dst] += message
        
        # Update atom features
        updated_features = []
        for i in range(num_atoms):
            # Concatenate original features with aggregated messages
            update_input = torch.cat([atom_features[i], messages[i]], dim=0)
            updated_feat = self.update_net(update_input)
            updated_features.append(updated_feat)
        
        return torch.stack(updated_features)

class EnhancedGNNModel(nn.Module):
    """Enhanced Graph Neural Network with custom message passing"""
    
    def __init__(self, atom_feat_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super(EnhancedGNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.atom_embedding = nn.Linear(atom_feat_dim, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, graph_data):
        """Forward pass through the enhanced GNN"""
        # Initial atom embeddings
        h = self.atom_embedding(graph_data.atom_features)
        
        # Message passing through graph convolution layers
        for layer in self.gnn_layers:
            h_new = layer(h, graph_data.edge_indices)
            h = F.relu(h_new) + h  # Residual connection
        
        # Global pooling (mean of all atom features)
        graph_repr = torch.mean(h, dim=0, keepdim=True)
        
        # Final prediction
        output = self.output_layers(graph_repr)
        
        return output

class EnhancedGNNPredictor:
    """Enhanced Graph Neural Network predictor with custom message passing"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_enhanced_gnn_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize molecular graph builder
        self.graph_builder = MolecularGraphBuilder()
        
        # Model storage
        self.models = {}
        self.training_data = {}
        self.reference_smiles = {}
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {self.device}")
    
    async def initialize_models(self, target: str = "EGFR"):
        """Initialize Enhanced GNN models for specific target"""
        logger.info(f"🎯 Initializing Enhanced GNN model for {target}")
        
        try:
            # Load or download training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) < 50:
                logger.warning(f"❌ Insufficient training data for Enhanced GNN: {len(training_data)} samples")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Try to load existing model
            model_file = self.model_dir / f"{target}_enhanced_gnn_model.pkl"
            
            if model_file.exists():
                try:
                    model_data = joblib.load(model_file)
                    self.models[target] = model_data
                    logger.info(f"✅ Loaded cached Enhanced GNN model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached Enhanced GNN model: {e}")
            
            # Train new Enhanced GNN model
            success = await self._train_enhanced_gnn_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing Enhanced GNN model for {target}: {e}")
            return False
    
    async def _train_enhanced_gnn_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Train Enhanced GNN model on ChEMBL molecular graphs"""
        logger.info(f"🧠 Training Enhanced GNN for {target} with {len(training_data)} compounds")
        
        try:
            # Convert SMILES to molecular graphs
            graphs = []
            targets = []
            
            logger.info("🔄 Converting SMILES to enhanced molecular graphs...")
            for i, (_, row) in enumerate(training_data.iterrows()):
                if i % 200 == 0:
                    logger.info(f"  Processed {i}/{len(training_data)} molecules...")
                
                graph = self.graph_builder.smiles_to_graph(row['smiles'])
                if graph is not None:
                    graphs.append(graph)
                    targets.append(row['pic50'])
            
            if len(graphs) < 50:
                logger.error(f"❌ Too few valid molecular graphs: {len(graphs)}")
                return False
            
            logger.info(f"📊 Created {len(graphs)} enhanced molecular graphs for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                graphs, targets, test_size=0.2, random_state=42
            )
            
            logger.info(f"📈 Train: {len(X_train)}, Test: {len(X_test)} molecular graphs")
            
            # Initialize Enhanced GNN model
            model = EnhancedGNNModel(
                atom_feat_dim=self.graph_builder.atom_feat_dim,
                hidden_dim=128,
                num_layers=3
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            logger.info(f"🏗️ Enhanced GNN Architecture:")
            logger.info(f"  🔬 Atom features: {self.graph_builder.atom_feat_dim}")
            logger.info(f"  🧠 Hidden dimension: 128")
            logger.info(f"  📚 Message passing layers: 3")
            logger.info(f"  🔄 Custom graph convolution with residual connections")
            
            # Training loop
            model.train()
            best_test_loss = float('inf')
            
            logger.info("🚀 Starting Enhanced GNN training...")
            
            for epoch in range(50):  # 50 epochs
                # Training
                model.train()
                train_losses = []
                
                for i, (graph, target) in enumerate(zip(X_train, y_train)):
                    graph = graph.to(self.device)
                    target_tensor = torch.FloatTensor([target]).unsqueeze(0).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(graph)
                    loss = criterion(output, target_tensor)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Evaluation every 10 epochs
                if epoch % 10 == 0:
                    model.eval()
                    test_losses = []
                    test_preds = []
                    test_targets = []
                    
                    with torch.no_grad():
                        for graph, target in zip(X_test, y_test):
                            graph = graph.to(self.device)
                            output = model(graph)
                            
                            target_tensor = torch.FloatTensor([target]).unsqueeze(0).to(self.device)
                            loss = criterion(output, target_tensor)
                            
                            test_losses.append(loss.item())
                            test_preds.append(output.cpu().numpy()[0][0])
                            test_targets.append(target)
                    
                    avg_train_loss = np.mean(train_losses)
                    avg_test_loss = np.mean(test_losses)
                    
                    # Calculate R²
                    test_r2 = r2_score(test_targets, test_preds)
                    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
                    
                    logger.info(f"  Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}, R²={test_r2:.3f}, RMSE={test_rmse:.3f}")
                    
                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
            
            # Final evaluation
            logger.info("📊 Final evaluation...")
            model.eval()
            
            # Training predictions
            train_preds = []
            train_targets = []
            
            with torch.no_grad():
                for graph, target in zip(X_train, y_train):
                    graph = graph.to(self.device)
                    output = model(graph)
                    train_preds.append(output.cpu().numpy()[0][0])
                    train_targets.append(target)
            
            # Test predictions
            test_preds = []
            test_targets = []
            
            with torch.no_grad():
                for graph, target in zip(X_test, y_test):
                    graph = graph.to(self.device)
                    output = model(graph)
                    test_preds.append(output.cpu().numpy()[0][0])
                    test_targets.append(target)
            
            # Calculate final metrics
            train_r2 = r2_score(train_targets, train_preds)
            test_r2 = r2_score(test_targets, test_preds)
            test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
            
            logger.info(f"🎯 Enhanced GNN Final Performance:")
            logger.info(f"  🎯 Train R²: {train_r2:.3f}")
            logger.info(f"  🎯 Test R²: {test_r2:.3f}")
            logger.info(f"  📏 Test RMSE: {test_rmse:.3f}")
            
            # Save model
            model_data = {
                'model': model.cpu(),  # Move to CPU for saving
                'target': target,
                'model_type': 'enhanced_gnn',
                'training_size': len(training_data),
                'performance': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                },
                'device': str(self.device),
                'architecture': 'Enhanced GNN with Custom Message Passing'
            }
            
            self.models[target] = model_data
            
            model_file = self.model_dir / f"{target}_enhanced_gnn_model.pkl"
            joblib.dump(model_data, model_file)
            
            logger.info(f"✅ Enhanced GNN model saved for {target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training Enhanced GNN model for {target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def predict_ic50_gnn(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 using trained Enhanced GNN model"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing Enhanced GNN model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            logger.error(f"❌ No Enhanced GNN model available for {target}")
            return {
                'error': f'No Enhanced GNN model available for {target}',
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
        
        try:
            # Convert SMILES to molecular graph
            graph = self.graph_builder.smiles_to_graph(smiles)
            if graph is None:
                return {
                    'error': 'Invalid SMILES string - could not create molecular graph',
                    'pic50': None,
                    'ic50_nm': None,
                    'confidence': 0.0,
                    'similarity': 0.0,
                    'model_type': 'error'
                }
            
            # Load model and make prediction
            model_data = self.models[target]
            model = model_data['model'].to(self.device)
            model.eval()
            
            with torch.no_grad():
                # Move graph to device
                graph = graph.to(self.device)
                prediction = model(graph)
                predicted_pic50 = prediction.cpu().numpy()[0][0]
            
            # Convert to IC50 in nM
            ic50_nm = 10 ** (9 - predicted_pic50)
            
            # Calculate similarity to training set
            similarity = chembl_manager.calculate_tanimoto_similarity(
                smiles, self.reference_smiles.get(target, [])
            )
            
            # Calculate confidence based on similarity and model performance
            base_confidence = model_data['performance']['test_r2']
            similarity_weight = similarity * 0.9 + 0.1  # High weight for GNN confidence
            confidence = min(base_confidence * similarity_weight, 1.0)
            
            return {
                'pic50': float(predicted_pic50),
                'ic50_nm': float(ic50_nm),
                'confidence': float(confidence),
                'similarity': float(similarity),
                'model_type': 'enhanced_gnn',
                'target_specific': True,
                'architecture': 'Enhanced GNN with Custom Message Passing',
                'model_performance': model_data['performance'],
                'training_size': model_data['training_size'],
                'graph_features': {
                    'atom_features': self.graph_builder.atom_feat_dim,
                    'message_passing_layers': 3,
                    'residual_connections': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in Enhanced GNN prediction: {e}")
            return {
                'error': str(e),
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets"""
        return chembl_manager.get_available_targets()
    
    def get_model_info(self, target: str) -> Dict:
        """Get information about trained Enhanced GNN model"""
        if target in self.models:
            return {
                'target': target,
                'available': True,
                'model_type': 'enhanced_gnn',
                'architecture': 'Enhanced GNN with Custom Message Passing',
                'performance': self.models[target]['performance'],
                'training_size': self.models[target]['training_size'],
                'graph_features': {
                    'atom_features': self.graph_builder.atom_feat_dim,
                    'message_passing_layers': 3,
                    'residual_connections': True
                }
            }
        else:
            return {
                'target': target,
                'available': False,
                'model_type': 'enhanced_gnn',
                'architecture': 'Enhanced GNN with Custom Message Passing',
                'performance': None,
                'training_size': 0
            }

# Global Enhanced GNN predictor instance
enhanced_gnn_predictor = EnhancedGNNPredictor()