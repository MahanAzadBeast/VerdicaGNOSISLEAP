#!/usr/bin/env python3
"""
Optimize Existing SMILES Integration
Instead of restarting, improve the existing 21% SMILES coverage
using smarter local matching with the ChEMBL database we already have
"""

import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SMILESOptimizer:
    """Optimizes existing SMILES integration without API calls"""
    
    def __init__(self):
        self.github_dir = Path("clinical_trial_dataset/data/github_final")
        
        # Enhanced drug name mappings (no API calls needed)
        self.comprehensive_mappings = {
            # Common drug names
            'ropivacaine': ['ropivacaine', 'naropin', 'ropivacaine hydrochloride'],
            'botox': ['botulinum toxin type a', 'onabotulinumtoxina', 'clostridium botulinum toxin'],
            'acetaminophen': ['paracetamol', 'acetaminophen', 'tylenol'],
            'buprenorphine': ['buprenorphine', 'subutex', 'buprenorphine hydrochloride'],
            'iron sucrose': ['iron sucrose', 'iron(iii) sucrose', 'ferric sucrose'],
            'astragalus powder': ['astragalus', 'astragalus membranaceus'],
            'melatonin': ['melatonin', 'n-acetyl-5-methoxytryptamine'],
            
            # Anesthetics
            'lidocaine': ['lidocaine', 'lignocaine', 'xylocaine'],
            'procaine': ['procaine', 'novocaine'],
            'articaine': ['articaine', 'articaine hydrochloride'],
            
            # Antibiotics
            'amoxicillin': ['amoxicillin', 'amoxycillin'],
            'ciprofloxacin': ['ciprofloxacin', 'cipro'],
            'doxycycline': ['doxycycline', 'vibramycin'],
            
            # Pain medications
            'morphine': ['morphine', 'morphine sulfate'],
            'fentanyl': ['fentanyl', 'fentanyl citrate'],
            'tramadol': ['tramadol', 'tramadol hydrochloride'],
            
            # Common drugs
            'metformin': ['metformin', 'metformin hydrochloride'],
            'insulin': ['insulin', 'human insulin'],
            'warfarin': ['warfarin', 'warfarin sodium', 'coumadin']
        }
    
    def load_existing_data(self):
        """Load existing SMILES integration and ChEMBL data"""
        logger.info("📂 LOADING EXISTING DATA")
        logger.info("=" * 50)
        
        # Load existing SMILES integration
        smiles_file = self.github_dir / "clinical_trials_with_smiles_complete.csv"
        chembl_file = self.github_dir / "chembl_complete_dataset.csv"
        
        smiles_df = pd.DataFrame()
        chembl_df = pd.DataFrame()
        
        if smiles_file.exists():
            smiles_df = pd.read_csv(smiles_file, low_memory=False)
            logger.info(f"✅ Existing SMILES integration: {len(smiles_df):,} trials")
            
            current_coverage = smiles_df['smiles'].notna().sum()
            coverage_pct = (current_coverage / len(smiles_df)) * 100
            logger.info(f"   Current SMILES coverage: {current_coverage:,} ({coverage_pct:.1f}%)")
        
        if chembl_file.exists():
            chembl_df = pd.read_csv(chembl_file)
            logger.info(f"✅ ChEMBL database: {len(chembl_df):,} compounds")
        
        return smiles_df, chembl_df
    
    def create_enhanced_chembl_lookup(self, chembl_df):
        """Create enhanced ChEMBL lookup with multiple name variations"""
        logger.info("🔍 Creating enhanced ChEMBL lookup...")
        
        enhanced_lookup = {}
        
        for _, row in chembl_df.iterrows():
            drug_name = str(row['primary_drug']).lower().strip()
            smiles_data = {
                'smiles': row['smiles'],
                'chembl_id': row['chembl_id'],
                'molecular_weight': row.get('mol_molecular_weight'),
                'logp': row.get('mol_logp'),
                'source': 'local_chembl'
            }
            
            # Add primary name
            enhanced_lookup[drug_name] = smiles_data
            
            # Add variations
            variations = self._generate_name_variations(drug_name)
            for variation in variations:
                if variation not in enhanced_lookup:  # Don't overwrite
                    enhanced_lookup[variation] = smiles_data
        
        # Add manual mappings
        for base_name, alternatives in self.comprehensive_mappings.items():
            base_data = enhanced_lookup.get(base_name)
            if not base_data:
                # Try to find it under alternatives
                for alt in alternatives:
                    if alt in enhanced_lookup:
                        base_data = enhanced_lookup[alt]
                        break
            
            if base_data:
                # Add all alternatives
                for alt in alternatives:
                    enhanced_lookup[alt] = base_data
        
        logger.info(f"✅ Enhanced lookup created: {len(enhanced_lookup):,} drug name variations")
        return enhanced_lookup
    
    def _generate_name_variations(self, drug_name):
        """Generate name variations for better matching"""
        variations = set()
        
        # Original name
        variations.add(drug_name)
        
        # Without spaces
        variations.add(drug_name.replace(' ', ''))
        
        # Without hyphens
        variations.add(drug_name.replace('-', ''))
        variations.add(drug_name.replace('-', ' '))
        
        # Common salt removals
        salt_suffixes = ['hydrochloride', 'sulfate', 'sodium', 'potassium', 'phosphate']
        for suffix in salt_suffixes:
            if drug_name.endswith(f' {suffix}'):
                variations.add(drug_name.replace(f' {suffix}', ''))
        
        # First word only
        if ' ' in drug_name:
            variations.add(drug_name.split()[0])
        
        return list(variations)
    
    def optimize_smiles_matching(self, trials_df, enhanced_lookup):
        """Optimize SMILES matching using enhanced lookup"""
        logger.info("🔧 OPTIMIZING SMILES MATCHING WITH LOCAL DATA")
        
        optimized_trials = []
        original_smiles_count = trials_df['smiles'].notna().sum()
        new_matches = 0
        
        for _, trial in trials_df.iterrows():
            optimized_trial = trial.to_dict()
            
            # If already has SMILES, keep it
            if pd.notna(trial.get('smiles')):
                optimized_trials.append(optimized_trial)
                continue
            
            # Try to find SMILES for trials without
            drug_name = trial.get('primary_drug')
            
            if drug_name and pd.notna(drug_name):
                clean_drug = self._clean_drug_name(str(drug_name))
                
                if clean_drug:
                    # Try enhanced lookup
                    smiles_data = self._enhanced_lookup_search(clean_drug, enhanced_lookup)
                    
                    if smiles_data:
                        # Add SMILES data
                        optimized_trial.update({
                            'smiles': smiles_data['smiles'],
                            'smiles_source': smiles_data['chembl_id'],
                            'smiles_database': smiles_data['source'],
                            'smiles_search_method': 'enhanced_local_lookup',
                            'has_real_smiles': True,
                            'molecular_weight': smiles_data.get('molecular_weight'),
                            'logp': smiles_data.get('logp')
                        })
                        new_matches += 1
            
            optimized_trials.append(optimized_trial)
        
        optimized_df = pd.DataFrame(optimized_trials)
        
        # Statistics
        final_smiles_count = optimized_df['smiles'].notna().sum()
        improvement = final_smiles_count - original_smiles_count
        final_coverage = (final_smiles_count / len(optimized_df)) * 100
        
        logger.info(f"✅ Optimization complete:")
        logger.info(f"   Original SMILES: {original_smiles_count:,}")
        logger.info(f"   New matches found: {improvement:,}")
        logger.info(f"   Final SMILES: {final_smiles_count:,}")
        logger.info(f"   Final coverage: {final_coverage:.1f}%")
        
        return optimized_df
    
    def _enhanced_lookup_search(self, drug_name, enhanced_lookup):
        """Enhanced lookup search with multiple strategies"""
        drug_lower = drug_name.lower()
        
        # Direct lookup
        if drug_lower in enhanced_lookup:
            return enhanced_lookup[drug_lower]
        
        # Fuzzy matching
        for lookup_name, data in enhanced_lookup.items():
            if self._fuzzy_match(drug_lower, lookup_name):
                return data
        
        return None
    
    def _fuzzy_match(self, name1, name2):
        """Smart fuzzy matching for drug names"""
        # Exact match
        if name1 == name2:
            return True
        
        # Length-based substring matching
        if len(name1) > 4 and len(name2) > 4:
            if name1 in name2 or name2 in name1:
                return True
        
        # Word-based matching
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if words1 and words2:
            # Significant word overlap
            overlap = words1.intersection(words2)
            min_words = min(len(words1), len(words2))
            
            if len(overlap) >= min_words * 0.8:  # 80% word overlap
                return True
        
        return False
    
    def _clean_drug_name(self, drug_name):
        """Clean drug name for better matching"""
        clean = str(drug_name).strip().lower()
        
        # Remove dosage info
        clean = re.sub(r'\s+\d+\s*(mg|mcg|ml|g|%)\b.*', '', clean)
        
        # Remove formulation info  
        clean = re.sub(r'\s+(tablet|capsule|injection|solution|gel|cream)\b.*', '', clean)
        
        # Remove brand markers
        clean = re.sub(r'[®™©]', '', clean)
        
        # Remove parentheses
        clean = re.sub(r'\s*\([^)]*\)\s*', ' ', clean)
        
        # Clean spaces
        clean = ' '.join(clean.split())
        
        return clean if len(clean) > 2 else None
    
    def create_final_optimized_dataset(self):
        """Create final optimized dataset without restarting"""
        logger.info("🎯 CREATING FINAL OPTIMIZED DATASET")
        logger.info("✅ Using existing data + smart optimization")
        logger.info("🚫 NO MORE RESTARTS")
        logger.info("=" * 60)
        
        # Load existing data
        trials_df, chembl_df = self.load_existing_data()
        
        if trials_df.empty:
            logger.error("❌ No existing SMILES data found")
            return None
        
        # Create enhanced lookup
        enhanced_lookup = self.create_enhanced_chembl_lookup(chembl_df)
        
        # Optimize matching
        optimized_df = self.optimize_smiles_matching(trials_df, enhanced_lookup)
        
        # Save optimized dataset
        self.save_optimized_dataset(optimized_df)
        
        return optimized_df
    
    def save_optimized_dataset(self, df):
        """Save optimized dataset"""
        logger.info("💾 SAVING OPTIMIZED DATASET")
        
        # Save complete optimized dataset
        complete_file = self.github_dir / "clinical_trials_optimized_smiles.csv"
        df.to_csv(complete_file, index=False)
        
        # Create splits
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = self.github_dir / "train_optimized_smiles.csv"
        val_file = self.github_dir / "val_optimized_smiles.csv"
        test_file = self.github_dir / "test_optimized_smiles.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Final verification
        total_with_smiles = df['smiles'].notna().sum()
        smiles_coverage = (total_with_smiles / len(df)) * 100
        
        # Verify NCT02688101
        nct_check = df[df['nct_id'] == 'NCT02688101']
        nct_has_smiles = False
        if len(nct_check) > 0:
            nct_record = nct_check.iloc[0]
            nct_has_smiles = pd.notna(nct_record.get('smiles'))
        
        # Check file sizes
        logger.info(f"💾 Optimized datasets saved:")
        for file_path in [complete_file, train_file, val_file, test_file]:
            size_mb = os.path.getsize(file_path) / (1024*1024)
            status = "✅ GitHub OK" if size_mb < 100 else "❌ Too large"
            count = len(pd.read_csv(file_path))
            logger.info(f"   {status} {file_path.name}: {count:,} trials ({size_mb:.1f} MB)")
        
        logger.info(f"\\n📊 FINAL OPTIMIZED RESULTS:")
        logger.info(f"   Total trials: {len(df):,}")
        logger.info(f"   Trials with SMILES: {total_with_smiles:,} ({smiles_coverage:.1f}%)")
        logger.info(f"   NCT02688101 with SMILES: {'✅ YES' if nct_has_smiles else '❌ NO'}")
        
        return {
            'coverage': smiles_coverage,
            'total_with_smiles': total_with_smiles,
            'nct02688101_included': nct_has_smiles
        }

def main():
    """Main execution - NO MORE RESTARTS"""
    logger.info("🛑 STOPPING THE RESTART CYCLE")
    logger.info("🎯 OPTIMIZING EXISTING SMILES DATA")
    logger.info("✅ Working with what we have + smart improvements")
    logger.info("=" * 70)
    
    optimizer = SMILESOptimizer()
    
    # Create optimized dataset
    optimized_df = optimizer.create_final_optimized_dataset()
    
    if optimized_df is not None:
        logger.info("\\n" + "=" * 70)
        logger.info("🎉 SMILES OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info("✅ Used existing data intelligently")
        logger.info("✅ Enhanced matching algorithms")
        logger.info("✅ No API calls or long processes")
        logger.info("✅ Practical solution that works")
        logger.info("🚫 NO MORE RESTARTS NEEDED")
        
        return optimized_df
    else:
        logger.error("❌ Optimization failed")
        return None

if __name__ == "__main__":
    main()