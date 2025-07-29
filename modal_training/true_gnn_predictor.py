"""
True Graph Neural Network for Molecular Property Prediction
Implements actual GNN with molecular graphs, message passing, and graph convolutions
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
import dgl
from dgl.nn import GraphConv, GlobalMeanPooling
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from chembl_data_manager import chembl_manager

logger = logging.getLogger(__name__)

class MolecularGraphBuilder:
    """Build molecular graphs from SMILES for GNN processing"""
    
    # Atom feature mapping
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
        'num_hs': [0, 1, 2, 3, 4]
    }
    
    # Bond feature mapping
    BOND_FEATURES = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
        'conjugated': [False, True],
        'in_ring': [False, True],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
    }
    
    def __init__(self):
        # Calculate feature dimensions
        self.atom_feat_dim = sum(len(feat_list) for feat_list in self.ATOM_FEATURES.values())
        self.bond_feat_dim = sum(len(feat_list) for feat_list in self.BOND_FEATURES.values())
        
        logger.info(f"🔬 Atom feature dimension: {self.atom_feat_dim}")
        logger.info(f"🔗 Bond feature dimension: {self.bond_feat_dim}")
    
    def _safe_one_hot(self, value, choices):
        """Safe one-hot encoding with fallback for unknown values"""
        try:
            return [float(value == choice) for choice in choices]
        except:
            return [0.0] * len(choices)
    
    def _get_atom_features(self, atom):
        """Extract atom features for GNN"""
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
        
        return features
    
    def _get_bond_features(self, bond):
        """Extract bond features for GNN"""
        features = []
        
        # Bond type
        features.extend(self._safe_one_hot(bond.GetBondType(), self.BOND_FEATURES['bond_type']))
        
        # Conjugated
        features.extend(self._safe_one_hot(bond.GetIsConjugated(), self.BOND_FEATURES['conjugated']))
        
        # In ring
        features.extend(self._safe_one_hot(bond.IsInRing(), self.BOND_FEATURES['in_ring']))
        
        # Stereo
        features.extend(self._safe_one_hot(bond.GetStereo(), self.BOND_FEATURES['stereo']))
        
        return features
    
    def smiles_to_graph(self, smiles: str) -> Optional[dgl.DGLGraph]:
        """Convert SMILES to DGL molecular graph"""
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
            
            # Extract bonds and bond features
            edges = []
            bond_features = []
            
            for bond in mol.GetBonds():
                # DGL expects undirected graphs, so add both directions
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                bond_feat = self._get_bond_features(bond)
                
                edges.extend([(i, j), (j, i)])
                bond_features.extend([bond_feat, bond_feat])  # Same features for both directions
            
            # Handle single-atom molecules
            if len(edges) == 0:
                # Create self-loop for single atoms
                edges = [(0, 0)]
                bond_features = [[0.0] * self.bond_feat_dim]
            
            # Create DGL graph
            src, dst = zip(*edges)
            graph = dgl.graph((src, dst))
            
            # Add node features
            graph.ndata['feat'] = torch.FloatTensor(atom_features)
            
            # Add edge features  
            graph.edata['feat'] = torch.FloatTensor(bond_features)
            
            return graph
            
        except Exception as e:
            logger.warning(f"Error converting SMILES to graph: {smiles} - {e}")
            return None

class TrueGNNModel(nn.Module):
    """True Graph Neural Network with message passing and graph convolutions"""
    
    def __init__(self, atom_feat_dim: int, bond_feat_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super(TrueGNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.atom_embedding = nn.Linear(atom_feat_dim, hidden_dim)
        self.bond_embedding = nn.Linear(bond_feat_dim, hidden_dim)
        
        # Graph convolution layers with message passing
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        
        # Global pooling
        self.global_pool = GlobalMeanPooling()
        
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
    
    def forward(self, graphs):
        """Forward pass through the GNN"""
        # Node features
        h = graphs.ndata['feat']
        h = self.atom_embedding(h)
        
        # Message passing through graph convolution layers
        for layer in self.gnn_layers:
            h = layer(graphs, h)
        
        # Global pooling to get graph-level representation
        graphs.ndata['h'] = h
        graph_repr = self.global_pool(graphs, h)
        
        # Final prediction
        output = self.output_layers(graph_repr)
        
        return output

class TrueGNNPredictor:
    """True Graph Neural Network predictor using molecular graphs"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_true_gnn_models"):
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
        """Initialize True GNN models for specific target"""
        logger.info(f"🎯 Initializing True GNN model for {target}")
        
        try:
            # Load or download training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) < 50:
                logger.warning(f"❌ Insufficient training data for True GNN: {len(training_data)} samples")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Try to load existing model
            model_file = self.model_dir / f"{target}_true_gnn_model.pkl"
            
            if model_file.exists():
                try:
                    model_data = joblib.load(model_file)
                    self.models[target] = model_data
                    logger.info(f"✅ Loaded cached True GNN model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached True GNN model: {e}")
            
            # Train new True GNN model
            success = await self._train_true_gnn_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing True GNN model for {target}: {e}")
            return False
    
    async def _train_true_gnn_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Train True GNN model on ChEMBL molecular graphs"""
        logger.info(f"🧠 Training True GNN for {target} with {len(training_data)} compounds")
        
        try:
            # Convert SMILES to molecular graphs
            graphs = []
            targets = []
            
            logger.info("🔄 Converting SMILES to molecular graphs...")
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
            
            logger.info(f"📊 Created {len(graphs)} molecular graphs for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                graphs, targets, test_size=0.2, random_state=42
            )
            
            logger.info(f"📈 Train: {len(X_train)}, Test: {len(X_test)} molecular graphs")
            
            # Initialize True GNN model
            model = TrueGNNModel(
                atom_feat_dim=self.graph_builder.atom_feat_dim,
                bond_feat_dim=self.graph_builder.bond_feat_dim,
                hidden_dim=128,
                num_layers=3
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            logger.info(f"🏗️ True GNN Architecture:")
            logger.info(f"  🔬 Atom features: {self.graph_builder.atom_feat_dim}")
            logger.info(f"  🔗 Bond features: {self.graph_builder.bond_feat_dim}")
            logger.info(f"  🧠 Hidden dimension: 128")
            logger.info(f"  📚 Graph convolution layers: 3")
            
            # Training loop
            model.train()
            best_test_loss = float('inf')
            
            logger.info("🚀 Starting True GNN training...")
            
            for epoch in range(50):  # 50 epochs for True GNN
                # Training
                model.train()
                train_losses = []
                
                # Process in batches
                batch_size = 32
                for i in range(0, len(X_train), batch_size):
                    batch_graphs = X_train[i:i + batch_size]
                    batch_targets = torch.FloatTensor(y_train[i:i + batch_size]).unsqueeze(1).to(self.device)
                    
                    # Create batched graph
                    batched_graph = dgl.batch(batch_graphs).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batched_graph)
                    loss = criterion(outputs, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Evaluation
                if epoch % 10 == 0:
                    model.eval()
                    test_losses = []
                    all_test_preds = []
                    all_test_targets = []
                    
                    with torch.no_grad():
                        for i in range(0, len(X_test), batch_size):
                            batch_graphs = X_test[i:i + batch_size]
                            batch_targets = torch.FloatTensor(y_test[i:i + batch_size]).unsqueeze(1).to(self.device)
                            
                            batched_graph = dgl.batch(batch_graphs).to(self.device)
                            outputs = model(batched_graph)
                            loss = criterion(outputs, batch_targets)
                            
                            test_losses.append(loss.item())
                            all_test_preds.extend(outputs.cpu().numpy().flatten())
                            all_test_targets.extend(batch_targets.cpu().numpy().flatten())
                    
                    avg_train_loss = np.mean(train_losses)
                    avg_test_loss = np.mean(test_losses)
                    
                    # Calculate R²
                    test_r2 = r2_score(all_test_targets, all_test_preds)
                    test_rmse = np.sqrt(mean_squared_error(all_test_targets, all_test_preds))
                    
                    logger.info(f"  Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}, R²={test_r2:.3f}, RMSE={test_rmse:.3f}")
                    
                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
            
            # Final evaluation
            logger.info("📊 Final evaluation...")
            model.eval()
            
            # Evaluate on training set
            train_preds = []
            train_targets = []
            
            with torch.no_grad():
                for i in range(0, len(X_train), batch_size):
                    batch_graphs = X_train[i:i + batch_size]
                    batch_targets = y_train[i:i + batch_size]
                    
                    batched_graph = dgl.batch(batch_graphs).to(self.device)
                    outputs = model(batched_graph)
                    
                    train_preds.extend(outputs.cpu().numpy().flatten())
                    train_targets.extend(batch_targets)
            
            # Evaluate on test set  
            test_preds = []
            test_targets = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch_graphs = X_test[i:i + batch_size]
                    batch_targets = y_test[i:i + batch_size]
                    
                    batched_graph = dgl.batch(batch_graphs).to(self.device)
                    outputs = model(batched_graph)
                    
                    test_preds.extend(outputs.cpu().numpy().flatten())
                    test_targets.extend(batch_targets)
            
            # Calculate final metrics
            train_r2 = r2_score(train_targets, train_preds)
            test_r2 = r2_score(test_targets, test_preds)
            test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
            
            logger.info(f"🎯 True GNN Final Performance:")
            logger.info(f"  🎯 Train R²: {train_r2:.3f}")
            logger.info(f"  🎯 Test R²: {test_r2:.3f}")
            logger.info(f"  📏 Test RMSE: {test_rmse:.3f}")
            
            # Save model
            model_data = {
                'model': model.cpu(),  # Move to CPU for saving
                'target': target,
                'model_type': 'true_gnn',
                'training_size': len(training_data),
                'performance': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                },
                'device': str(self.device),
                'architecture': 'Graph Neural Network with Message Passing'
            }
            
            self.models[target] = model_data
            
            model_file = self.model_dir / f"{target}_true_gnn_model.pkl"
            joblib.dump(model_data, model_file)
            
            logger.info(f"✅ True GNN model saved for {target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training True GNN model for {target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def predict_ic50_gnn(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 using trained True GNN model"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing True GNN model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            logger.error(f"❌ No True GNN model available for {target}")
            return {
                'error': f'No True GNN model available for {target}',
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
            similarity_weight = similarity * 0.9 + 0.1  # Higher weight for GNN confidence
            confidence = min(base_confidence * similarity_weight, 1.0)
            
            return {
                'pic50': float(predicted_pic50),
                'ic50_nm': float(ic50_nm),
                'confidence': float(confidence),
                'similarity': float(similarity),
                'model_type': 'true_gnn',
                'target_specific': True,
                'architecture': 'Graph Neural Network with Message Passing',
                'model_performance': model_data['performance'],
                'training_size': model_data['training_size'],
                'graph_features': {
                    'atom_features': self.graph_builder.atom_feat_dim,
                    'bond_features': self.graph_builder.bond_feat_dim,
                    'message_passing_layers': 3
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in True GNN prediction: {e}")
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
        """Get information about trained True GNN model"""
        if target in self.models:
            return {
                'target': target,
                'available': True,
                'model_type': 'true_gnn',
                'architecture': 'Graph Neural Network with Message Passing',
                'performance': self.models[target]['performance'],
                'training_size': self.models[target]['training_size'],
                'graph_features': {
                    'atom_features': self.graph_builder.atom_feat_dim,
                    'bond_features': self.graph_builder.bond_feat_dim,
                    'message_passing_layers': 3
                }
            }
        else:
            return {
                'target': target,
                'available': False,
                'model_type': 'true_gnn',
                'architecture': 'Graph Neural Network with Message Passing',
                'performance': None,
                'training_size': 0
            }

# Global True GNN predictor instance
true_gnn_predictor = TrueGNNPredictor()