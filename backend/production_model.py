"""
Production Model 2 architecture for backend inference
"""

import torch
import torch.nn as nn

class ProductionCytotoxicityModel(nn.Module):
    """
    Production Model 2 architecture optimized for cancer cell line cytotoxicity
    """
    
    def __init__(self, molecular_dim=20, genomic_dim=25, hidden_dim=128):
        super().__init__()
        
        # Molecular feature processing
        self.molecular_encoder = nn.Sequential(
            nn.Linear(molecular_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Genomic feature processing  
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combined prediction head
        combined_dim = hidden_dim + hidden_dim//2
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim//2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, molecular_features, genomic_features):
        # Process features separately
        mol_features = self.molecular_encoder(molecular_features)
        gen_features = self.genomic_encoder(genomic_features)
        
        # Combine and predict
        combined = torch.cat([mol_features, gen_features], dim=1)
        return self.prediction_head(combined)