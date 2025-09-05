"""
Name-to-Structure Mapping Utilities
Handles drug name standardization and structure mapping
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class DrugNameMapper:
    """
    Maps drug names to standardized forms and handles name variations
    """
    
    def __init__(self):
        # Common drug name mappings
        self.standard_mappings = {
            # Pain medications
            'acetaminophen': 'paracetamol',
            'acetaminophen/apap': 'paracetamol',
            'tylenol': 'paracetamol',
            
            # Cardiovascular
            'epinephrine': 'adrenaline',
            'norepinephrine': 'noradrenaline',
            
            # Antibiotics
            'amoxicillin': 'amoxycillin',
            
            # Anesthetics
            'lidocaine': 'lignocaine',
            'procaine': 'novocaine',
            
            # Biologics
            'botox': 'botulinum toxin type a',
            'restylane': 'hyaluronic acid',
            
            # Neurological
            'levodopa': 'l-dopa',
        }
        
        # Common salt and formulation suffixes
        self.salt_suffixes = {
            'hydrochloride', 'hcl', 'sulfate', 'sodium', 'potassium',
            'calcium', 'phosphate', 'acetate', 'citrate', 'tartrate',
            'maleate', 'succinate', 'fumarate', 'oxalate', 'mesylate',
            'besylate', 'tosylate'
        }
        
        # Formulation terms to remove
        self.formulation_terms = {
            'tablet', 'capsule', 'injection', 'solution', 'gel', 'cream',
            'patch', 'spray', 'powder', 'suspension', 'infusion', 'drops',
            'oral', 'iv', 'im', 'sc', 'topical', 'vaginal', 'rectal'
        }
    
    def clean_drug_name(self, drug_name: str) -> Optional[str]:
        """
        Clean and standardize drug name
        
        Args:
            drug_name: Raw drug name from source
            
        Returns:
            Cleaned drug name or None if invalid
        """
        if not drug_name or pd.isna(drug_name):
            return None
        
        clean = str(drug_name).strip().lower()
        
        # Remove dosage information
        clean = re.sub(r'\s*\d+\s*(mg|mcg|ml|g|%|unit|iu|mmol)\b.*', '', clean, flags=re.IGNORECASE)
        
        # Remove formulation information
        formulation_pattern = '|'.join(self.formulation_terms)
        clean = re.sub(f'\\s+({formulation_pattern})\\b.*', '', clean, flags=re.IGNORECASE)
        
        # Remove brand indicators
        clean = re.sub(r'[®™©]', '', clean)
        
        # Remove parentheses and brackets
        clean = re.sub(r'\s*\([^)]*\)\s*', ' ', clean)
        clean = re.sub(r'\s*\[[^\]]*\]\s*', ' ', clean)
        
        # Remove "based" and "containing" phrases
        clean = re.sub(r'\s+(based|containing|with|plus|and)\s+.*', '', clean, flags=re.IGNORECASE)
        
        # Clean up spaces and punctuation
        clean = re.sub(r'\s+', ' ', clean).strip()
        clean = clean.strip(' .-/')
        
        # Skip if too short or obviously not a drug
        if len(clean) < 3:
            return None
        
        # Skip procedures, devices, etc.
        skip_terms = {
            'procedure', 'surgery', 'therapy', 'treatment', 'intervention',
            'device', 'implant', 'catheter', 'stent', 'graft', 'radiation',
            'laser', 'ultrasound', 'imaging', 'placebo', 'control'
        }
        
        if any(term in clean for term in skip_terms):
            return None
        
        return clean
    
    def get_standard_name(self, drug_name: str) -> str:
        """
        Get standardized name using mappings
        
        Args:
            drug_name: Cleaned drug name
            
        Returns:
            Standardized drug name
        """
        if not drug_name:
            return drug_name
        
        drug_lower = drug_name.lower()
        
        # Direct mapping
        if drug_lower in self.standard_mappings:
            return self.standard_mappings[drug_lower]
        
        # Check for partial matches
        for original, standard in self.standard_mappings.items():
            if original in drug_lower or drug_lower in original:
                return standard
        
        return drug_name
    
    def remove_salt_suffix(self, drug_name: str) -> str:
        """
        Remove common salt suffixes from drug names
        
        Args:
            drug_name: Drug name potentially with salt suffix
            
        Returns:
            Drug name with salt suffix removed
        """
        if not drug_name:
            return drug_name
        
        drug_lower = drug_name.lower()
        
        for suffix in self.salt_suffixes:
            if drug_lower.endswith(f' {suffix}'):
                return drug_name[:-len(f' {suffix}')].strip()
        
        return drug_name
    
    def generate_name_variations(self, drug_name: str) -> List[str]:
        """
        Generate name variations for better matching
        
        Args:
            drug_name: Base drug name
            
        Returns:
            List of name variations to try
        """
        variations = set()
        
        if not drug_name:
            return []
        
        # Original name
        variations.add(drug_name)
        
        # Cleaned name
        cleaned = self.clean_drug_name(drug_name)
        if cleaned:
            variations.add(cleaned)
        
        # Standard name
        standard = self.get_standard_name(drug_name)
        variations.add(standard)
        
        # Without salt suffix
        no_salt = self.remove_salt_suffix(drug_name)
        variations.add(no_salt)
        
        # Without spaces
        variations.add(drug_name.replace(' ', ''))
        
        # Without hyphens
        variations.add(drug_name.replace('-', ''))
        variations.add(drug_name.replace('-', ' '))
        
        # First word only
        if ' ' in drug_name:
            variations.add(drug_name.split()[0])
        
        # Common salt additions
        base_name = self.remove_salt_suffix(drug_name)
        for salt in ['hydrochloride', 'sulfate', 'sodium']:
            variations.add(f"{base_name} {salt}")
        
        # Remove empty strings and short names
        variations = {v for v in variations if v and len(v) > 2}
        
        return sorted(list(variations))
    
    def fuzzy_match_score(self, name1: str, name2: str) -> float:
        """
        Calculate fuzzy match score between drug names
        
        Args:
            name1: First drug name
            name2: Second drug name
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not name1 or not name2:
            return 0.0
        
        name1_lower = name1.lower().strip()
        name2_lower = name2.lower().strip()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        # Substring match
        if name1_lower in name2_lower or name2_lower in name1_lower:
            longer = max(len(name1_lower), len(name2_lower))
            shorter = min(len(name1_lower), len(name2_lower))
            return shorter / longer
        
        # Word overlap
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        if words1 and words2:
            overlap = words1.intersection(words2)
            total_words = words1.union(words2)
            return len(overlap) / len(total_words)
        
        return 0.0
    
    def find_best_match(self, target_name: str, candidate_names: List[str], min_score: float = 0.7) -> Optional[str]:
        """
        Find best matching name from candidates
        
        Args:
            target_name: Name to match
            candidate_names: List of candidate names
            min_score: Minimum match score required
            
        Returns:
            Best matching name or None if no good match
        """
        if not target_name or not candidate_names:
            return None
        
        best_match = None
        best_score = 0.0
        
        # Try all variations of target name
        target_variations = self.generate_name_variations(target_name)
        
        for target_var in target_variations:
            for candidate in candidate_names:
                score = self.fuzzy_match_score(target_var, candidate)
                
                if score > best_score and score >= min_score:
                    best_match = candidate
                    best_score = score
        
        if best_match:
            logger.debug(f"Best match for '{target_name}': '{best_match}' (score: {best_score:.3f})")
        
        return best_match