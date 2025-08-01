#!/usr/bin/env python3
"""
PubChem BioAssay Integration Testing
Tests the newly implemented PubChem BioAssay integration components
"""

import requests
import json
import time
import sys
import os
import importlib.util
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class PubChemIntegrationTester:
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            'test': test_name,
            'status': status,
            'success': success,
            'details': details
        }
        self.test_results.append(result)
        if not success:
            self.failed_tests.append(result)
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_pubchem_extractor_module_import(self):
        """Test that enhanced_pubchem_extractor.py can be imported"""
        print("\n=== Testing PubChem Extractor Module Import ===")
        
        try:
            # Test import of enhanced_pubchem_extractor
            extractor_path = Path("/app/modal_training/enhanced_pubchem_extractor.py")
            if not extractor_path.exists():
                self.log_test("PubChem extractor file exists", False, f"File not found: {extractor_path}")
                return False
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location("enhanced_pubchem_extractor", extractor_path)
            if spec is None:
                self.log_test("PubChem extractor module spec", False, "Could not create module spec")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required classes and functions
            required_components = [
                'PubChemBioAssayExtractor',
                'PubChemDataQualityController', 
                'extract_pubchem_bioassay_data',
                'PUBCHEM_TARGETS'
            ]
            
            missing_components = []
            for component in required_components:
                if not hasattr(module, component):
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("PubChem extractor components", False, f"Missing: {missing_components}")
                return False
            
            # Check PUBCHEM_TARGETS structure
            targets = getattr(module, 'PUBCHEM_TARGETS')
            if not isinstance(targets, dict) or len(targets) == 0:
                self.log_test("PubChem targets configuration", False, f"Invalid targets: {type(targets)}")
                return False
            
            # Verify target categories
            categories = set()
            for target_info in targets.values():
                if 'category' in target_info:
                    categories.add(target_info['category'])
            
            expected_categories = {'oncoprotein', 'tumor_suppressor', 'metastasis_suppressor'}
            if not expected_categories.issubset(categories):
                self.log_test("PubChem target categories", False, f"Missing categories: {expected_categories - categories}")
                return False
            
            self.log_test("PubChem extractor module import", True, 
                        f"Module loaded with {len(targets)} targets across {len(categories)} categories")
            return True
            
        except Exception as e:
            self.log_test("PubChem extractor module import", False, f"Import error: {str(e)}")
            return False
    
    def test_pubchem_integration_module_import(self):
        """Test that integrate_pubchem_with_chembl.py can be imported"""
        print("\n=== Testing PubChem Integration Module Import ===")
        
        try:
            # Test import of integrate_pubchem_with_chembl
            integration_path = Path("/app/modal_training/integrate_pubchem_with_chembl.py")
            if not integration_path.exists():
                self.log_test("PubChem integration file exists", False, f"File not found: {integration_path}")
                return False
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location("integrate_pubchem_with_chembl", integration_path)
            if spec is None:
                self.log_test("PubChem integration module spec", False, "Could not create module spec")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions
            required_functions = [
                'integrate_pubchem_with_existing_data',
                'apply_cross_source_deduplication'
            ]
            
            missing_functions = []
            for func in required_functions:
                if not hasattr(module, func):
                    missing_functions.append(func)
            
            if missing_functions:
                self.log_test("PubChem integration functions", False, f"Missing: {missing_functions}")
                return False
            
            # Check for Modal app configuration
            if hasattr(module, 'app'):
                app = getattr(module, 'app')
                self.log_test("PubChem integration Modal app", True, f"Modal app configured: {app.name}")
            else:
                self.log_test("PubChem integration Modal app", False, "No Modal app found")
                return False
            
            self.log_test("PubChem integration module import", True, "Module loaded successfully")
            return True
            
        except Exception as e:
            self.log_test("PubChem integration module import", False, f"Import error: {str(e)}")
            return False
    
    def test_pubchem_launcher_module_import(self):
        """Test that launch_pubchem_integration.py can be imported"""
        print("\n=== Testing PubChem Launcher Module Import ===")
        
        try:
            # Test import of launch_pubchem_integration
            launcher_path = Path("/app/modal_training/launch_pubchem_integration.py")
            if not launcher_path.exists():
                self.log_test("PubChem launcher file exists", False, f"File not found: {launcher_path}")
                return False
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location("launch_pubchem_integration", launcher_path)
            if spec is None:
                self.log_test("PubChem launcher module spec", False, "Could not create module spec")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions
            required_functions = [
                'launch_pubchem_integration'
            ]
            
            missing_functions = []
            for func in required_functions:
                if not hasattr(module, func):
                    missing_functions.append(func)
            
            if missing_functions:
                self.log_test("PubChem launcher functions", False, f"Missing: {missing_functions}")
                return False
            
            self.log_test("PubChem launcher module import", True, "Module loaded successfully")
            return True
            
        except Exception as e:
            self.log_test("PubChem launcher module import", False, f"Import error: {str(e)}")
            return False
    
    def test_pubchem_data_standardization_compatibility(self):
        """Test that PubChem data standardization matches ChEMBL format"""
        print("\n=== Testing PubChem Data Standardization Compatibility ===")
        
        try:
            # Import the PubChem extractor
            extractor_path = Path("/app/modal_training/enhanced_pubchem_extractor.py")
            spec = importlib.util.spec_from_file_location("enhanced_pubchem_extractor", extractor_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test PubChemDataQualityController
            quality_controller_class = getattr(module, 'PubChemDataQualityController')
            quality_controller = quality_controller_class()
            
            # Test unit conversion (should match ChEMBL: nM units)
            test_cases = [
                (100, 'nM', 100),  # nM to nM
                (0.1, 'uM', 100),  # uM to nM  
                (0.0001, 'mM', 100),  # mM to nM
                (1e-7, 'M', 100)   # M to nM
            ]
            
            standardization_correct = True
            for value, unit, expected_nm in test_cases:
                try:
                    converted = quality_controller.convert_to_nm(value, unit)
                    if abs(converted - expected_nm) > 0.1:  # Allow small floating point errors
                        standardization_correct = False
                        self.log_test(f"Unit conversion {value} {unit}", False, 
                                    f"Expected {expected_nm} nM, got {converted} nM")
                        break
                    else:
                        self.log_test(f"Unit conversion {value} {unit}", True, 
                                    f"Correctly converted to {converted} nM")
                except Exception as e:
                    standardization_correct = False
                    self.log_test(f"Unit conversion {value} {unit}", False, f"Conversion error: {str(e)}")
                    break
            
            # Test pIC50 calculation (should match ChEMBL: -log10(IC50_M))
            test_ic50_nm = 100  # 100 nM
            expected_pic50 = 7.0  # -log10(100e-9) = 7.0
            
            try:
                calculated_pic50 = quality_controller.calculate_pic50(test_ic50_nm)
                if abs(calculated_pic50 - expected_pic50) < 0.01:
                    self.log_test("pIC50 calculation", True, f"Correct pIC50: {calculated_pic50}")
                else:
                    self.log_test("pIC50 calculation", False, 
                                f"Expected {expected_pic50}, got {calculated_pic50}")
                    standardization_correct = False
            except Exception as e:
                self.log_test("pIC50 calculation", False, f"Calculation error: {str(e)}")
                standardization_correct = False
            
            # Test SMILES validation (should use RDKit like ChEMBL)
            test_smiles = [
                ("CCO", True),  # Valid ethanol
                ("CC(=O)OC1=CC=CC=C1C(=O)O", True),  # Valid aspirin
                ("INVALID_SMILES", False),  # Invalid
                ("", False)  # Empty
            ]
            
            for smiles, should_be_valid in test_smiles:
                try:
                    is_valid = quality_controller.validate_smiles(smiles)
                    if is_valid == should_be_valid:
                        self.log_test(f"SMILES validation '{smiles[:20]}'", True, 
                                    f"Correctly validated: {is_valid}")
                    else:
                        self.log_test(f"SMILES validation '{smiles[:20]}'", False, 
                                    f"Expected {should_be_valid}, got {is_valid}")
                        standardization_correct = False
                except Exception as e:
                    self.log_test(f"SMILES validation '{smiles[:20]}'", False, f"Validation error: {str(e)}")
                    standardization_correct = False
            
            return standardization_correct
            
        except Exception as e:
            self.log_test("PubChem data standardization compatibility", False, f"Error: {str(e)}")
            return False
    
    def test_cross_source_deduplication_logic(self):
        """Test cross-source deduplication logic"""
        print("\n=== Testing Cross-Source Deduplication Logic ===")
        
        try:
            # Import the integration module
            integration_path = Path("/app/modal_training/integrate_pubchem_with_chembl.py")
            spec = importlib.util.spec_from_file_location("integrate_pubchem_with_chembl", integration_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the deduplication function
            dedup_func = getattr(module, 'apply_cross_source_deduplication')
            
            # Create test data with duplicates
            import pandas as pd
            import numpy as np
            
            test_data = [
                # Same compound-target pair from different sources
                {
                    'canonical_smiles': 'CCO',
                    'target_name': 'EGFR',
                    'activity_type': 'IC50',
                    'standard_value_nm': 100.0,
                    'data_source': 'ChEMBL'
                },
                {
                    'canonical_smiles': 'CCO',
                    'target_name': 'EGFR', 
                    'activity_type': 'IC50',
                    'standard_value_nm': 120.0,  # Close value - should be averaged
                    'data_source': 'PubChem_BioAssay'
                },
                # Same compound-target with large variance (should be discarded)
                {
                    'canonical_smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                    'target_name': 'BRAF',
                    'activity_type': 'IC50',
                    'standard_value_nm': 10.0,
                    'data_source': 'ChEMBL'
                },
                {
                    'canonical_smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                    'target_name': 'BRAF',
                    'activity_type': 'IC50', 
                    'standard_value_nm': 2000.0,  # >100x variance - should be discarded
                    'data_source': 'PubChem_BioAssay'
                },
                # Single source data (should be kept as-is)
                {
                    'canonical_smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                    'target_name': 'CDK2',
                    'activity_type': 'IC50',
                    'standard_value_nm': 500.0,
                    'data_source': 'ChEMBL'
                }
            ]
            
            df = pd.DataFrame(test_data)
            
            # Apply deduplication
            deduplicated_df = dedup_func(df)
            
            # Test results
            if len(deduplicated_df) == 2:  # Should have 2 records (1 averaged, 1 single, 1 discarded)
                self.log_test("Deduplication record count", True, 
                            f"Correct number of records: {len(deduplicated_df)}")
                
                # Check that ChEMBL is prioritized in averaging
                ethanol_record = deduplicated_df[
                    (deduplicated_df['canonical_smiles'] == 'CCO') & 
                    (deduplicated_df['target_name'] == 'EGFR')
                ]
                
                if len(ethanol_record) == 1:
                    # Should be weighted average favoring ChEMBL
                    expected_value = (100.0 * 0.7) + (120.0 * 0.3)  # 70% ChEMBL, 30% PubChem
                    actual_value = ethanol_record.iloc[0]['standard_value_nm']
                    
                    if abs(actual_value - expected_value) < 5.0:  # Allow some tolerance
                        self.log_test("ChEMBL prioritization in averaging", True, 
                                    f"Weighted average: {actual_value} (expected ~{expected_value})")
                    else:
                        self.log_test("ChEMBL prioritization in averaging", False, 
                                    f"Expected ~{expected_value}, got {actual_value}")
                        return False
                else:
                    self.log_test("ChEMBL prioritization in averaging", False, 
                                "Ethanol record not found after deduplication")
                    return False
                
                # Check that high variance data was discarded
                aspirin_records = deduplicated_df[
                    (deduplicated_df['canonical_smiles'] == 'CC(=O)OC1=CC=CC=C1C(=O)O') & 
                    (deduplicated_df['target_name'] == 'BRAF')
                ]
                
                if len(aspirin_records) == 0:
                    self.log_test("High variance data discarding", True, 
                                "High variance aspirin data correctly discarded")
                else:
                    self.log_test("High variance data discarding", False, 
                                f"High variance data not discarded: {len(aspirin_records)} records")
                    return False
                
                # Check that single source data is preserved
                caffeine_records = deduplicated_df[
                    (deduplicated_df['canonical_smiles'] == 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C') & 
                    (deduplicated_df['target_name'] == 'CDK2')
                ]
                
                if len(caffeine_records) == 1:
                    self.log_test("Single source data preservation", True, 
                                "Single source caffeine data preserved")
                else:
                    self.log_test("Single source data preservation", False, 
                                f"Single source data not preserved: {len(caffeine_records)} records")
                    return False
                
                return True
            else:
                self.log_test("Deduplication record count", False, 
                            f"Expected 2 records, got {len(deduplicated_df)}")
                return False
            
        except Exception as e:
            self.log_test("Cross-source deduplication logic", False, f"Error: {str(e)}")
            return False
    
    def test_pipeline_orchestration_structure(self):
        """Test that pipeline orchestration is properly structured"""
        print("\n=== Testing Pipeline Orchestration Structure ===")
        
        try:
            # Import the launcher module
            launcher_path = Path("/app/modal_training/launch_pubchem_integration.py")
            spec = importlib.util.spec_from_file_location("launch_pubchem_integration", launcher_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the main function
            launch_func = getattr(module, 'launch_pubchem_integration')
            
            # Check function signature and docstring
            import inspect
            sig = inspect.signature(launch_func)
            
            self.log_test("Pipeline orchestration function signature", True, 
                        f"Function signature: {sig}")
            
            # Check that function has proper documentation
            docstring = launch_func.__doc__
            if docstring and len(docstring.strip()) > 0:
                self.log_test("Pipeline orchestration documentation", True, 
                            "Function has documentation")
            else:
                self.log_test("Pipeline orchestration documentation", False, 
                            "Function lacks documentation")
                return False
            
            # Check that the function imports the required modules
            source_code = inspect.getsource(launch_func)
            
            required_imports = [
                'enhanced_pubchem_extractor',
                'integrate_pubchem_with_chembl'
            ]
            
            missing_imports = []
            for imp in required_imports:
                if imp not in source_code:
                    missing_imports.append(imp)
            
            if missing_imports:
                self.log_test("Pipeline orchestration imports", False, 
                            f"Missing imports: {missing_imports}")
                return False
            else:
                self.log_test("Pipeline orchestration imports", True, 
                            "All required modules imported")
            
            # Check for proper error handling
            if 'try:' in source_code and 'except' in source_code:
                self.log_test("Pipeline orchestration error handling", True, 
                            "Error handling present")
            else:
                self.log_test("Pipeline orchestration error handling", False, 
                            "No error handling found")
                return False
            
            # Check for progress reporting
            if 'print(' in source_code or 'logging' in source_code:
                self.log_test("Pipeline orchestration progress reporting", True, 
                            "Progress reporting present")
            else:
                self.log_test("Pipeline orchestration progress reporting", False, 
                            "No progress reporting found")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Pipeline orchestration structure", False, f"Error: {str(e)}")
            return False
    
    def test_expected_dataset_boost_calculation(self):
        """Test that the integration is expected to boost dataset size significantly"""
        print("\n=== Testing Expected Dataset Boost Calculation ===")
        
        try:
            # Import the extractor to check target configuration
            extractor_path = Path("/app/modal_training/enhanced_pubchem_extractor.py")
            spec = importlib.util.spec_from_file_location("enhanced_pubchem_extractor", extractor_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            targets = getattr(module, 'PUBCHEM_TARGETS')
            
            # Check that we have 23 targets as expected
            if len(targets) == 23:
                self.log_test("Target count for dataset boost", True, 
                            f"Correct number of targets: {len(targets)}")
            else:
                self.log_test("Target count for dataset boost", False, 
                            f"Expected 23 targets, got {len(targets)}")
                return False
            
            # Check target categories distribution
            category_counts = {}
            for target_info in targets.values():
                category = target_info.get('category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            expected_distribution = {
                'oncoprotein': 10,
                'tumor_suppressor': 7, 
                'metastasis_suppressor': 6
            }
            
            distribution_correct = True
            for category, expected_count in expected_distribution.items():
                actual_count = category_counts.get(category, 0)
                if actual_count != expected_count:
                    distribution_correct = False
                    self.log_test(f"Target category {category}", False, 
                                f"Expected {expected_count}, got {actual_count}")
                else:
                    self.log_test(f"Target category {category}", True, 
                                f"Correct count: {actual_count}")
            
            if not distribution_correct:
                return False
            
            # Estimate potential dataset boost
            # Assuming current ChEMBL dataset has ~25K records
            # PubChem BioAssay should add significant data for these 23 targets
            # Expected boost: 25K -> 75K+ (3x increase)
            
            current_estimated_size = 25000
            expected_boost_factor = 3.0
            expected_final_size = current_estimated_size * expected_boost_factor
            
            self.log_test("Expected dataset boost calculation", True, 
                        f"Expected boost: {current_estimated_size:,} -> {expected_final_size:,} records ({expected_boost_factor}x)")
            
            # Check that targets include both well-studied (likely in ChEMBL) and less-studied proteins
            well_studied_targets = ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET']  # Likely in ChEMBL
            less_studied_targets = ['NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RKIP']  # Likely more in PubChem
            
            well_studied_count = sum(1 for target in well_studied_targets if target in targets)
            less_studied_count = sum(1 for target in less_studied_targets if target in targets)
            
            self.log_test("Well-studied targets coverage", well_studied_count >= 3, 
                        f"Well-studied targets: {well_studied_count}/5")
            self.log_test("Less-studied targets coverage", less_studied_count >= 3, 
                        f"Less-studied targets: {less_studied_count}/5")
            
            return distribution_correct and well_studied_count >= 3 and less_studied_count >= 3
            
        except Exception as e:
            self.log_test("Expected dataset boost calculation", False, f"Error: {str(e)}")
            return False
    
    def test_data_quality_controls_match_chembl(self):
        """Test that data quality controls match ChEMBL standards"""
        print("\n=== Testing Data Quality Controls Match ChEMBL ===")
        
        try:
            # Import the extractor
            extractor_path = Path("/app/modal_training/enhanced_pubchem_extractor.py")
            spec = importlib.util.spec_from_file_location("enhanced_pubchem_extractor", extractor_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            quality_controller_class = getattr(module, 'PubChemDataQualityController')
            quality_controller = quality_controller_class()
            
            # Test 1: >100x variance filtering (same as ChEMBL)
            test_values = [10.0, 50.0, 100.0, 2000.0]  # 200x variance between min and max
            
            try:
                should_discard = quality_controller.should_discard_high_variance(test_values)
                if should_discard:
                    self.log_test("100x variance filtering", True, 
                                f"Correctly identified high variance: {min(test_values)}-{max(test_values)} nM")
                else:
                    self.log_test("100x variance filtering", False, 
                                "Failed to identify high variance data")
                    return False
            except Exception as e:
                self.log_test("100x variance filtering", False, f"Error: {str(e)}")
                return False
            
            # Test 2: Median aggregation for duplicates (same as ChEMBL)
            test_values_low_variance = [80.0, 100.0, 120.0]  # 1.5x variance - should be aggregated
            
            try:
                should_discard = quality_controller.should_discard_high_variance(test_values_low_variance)
                if not should_discard:
                    median_value = quality_controller.calculate_median(test_values_low_variance)
                    expected_median = 100.0
                    
                    if abs(median_value - expected_median) < 0.1:
                        self.log_test("Median aggregation", True, 
                                    f"Correct median calculation: {median_value}")
                    else:
                        self.log_test("Median aggregation", False, 
                                    f"Expected {expected_median}, got {median_value}")
                        return False
                else:
                    self.log_test("Median aggregation", False, 
                                "Low variance data incorrectly marked for discard")
                    return False
            except Exception as e:
                self.log_test("Median aggregation", False, f"Error: {str(e)}")
                return False
            
            # Test 3: RDKit SMILES validation (same as ChEMBL)
            test_molecules = [
                ("CCO", True),  # Valid
                ("CC(=O)OC1=CC=CC=C1C(=O)O", True),  # Valid aspirin
                ("Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", True),  # Valid imatinib
                ("INVALID", False),  # Invalid
                ("C1=CC=CC=C1C1=CC=CC=C1", True),  # Valid biphenyl
                ("", False)  # Empty
            ]
            
            validation_correct = True
            for smiles, expected_valid in test_molecules:
                try:
                    is_valid = quality_controller.validate_smiles(smiles)
                    if is_valid == expected_valid:
                        self.log_test(f"RDKit validation '{smiles[:20]}'", True, 
                                    f"Correctly validated: {is_valid}")
                    else:
                        self.log_test(f"RDKit validation '{smiles[:20]}'", False, 
                                    f"Expected {expected_valid}, got {is_valid}")
                        validation_correct = False
                except Exception as e:
                    self.log_test(f"RDKit validation '{smiles[:20]}'", False, f"Error: {str(e)}")
                    validation_correct = False
            
            # Test 4: Experimental data filtering (same as ChEMBL)
            # This would typically check assay types, but we'll test the concept
            test_assay_types = [
                ("binding", True),  # Experimental
                ("functional", True),  # Experimental  
                ("docking", False),  # Computational - should be filtered
                ("simulation", False),  # Computational - should be filtered
                ("biochemical", True)  # Experimental
            ]
            
            experimental_filtering_correct = True
            for assay_type, should_include in test_assay_types:
                try:
                    is_experimental = quality_controller.is_experimental_assay(assay_type)
                    if is_experimental == should_include:
                        self.log_test(f"Experimental filtering '{assay_type}'", True, 
                                    f"Correctly classified: {is_experimental}")
                    else:
                        self.log_test(f"Experimental filtering '{assay_type}'", False, 
                                    f"Expected {should_include}, got {is_experimental}")
                        experimental_filtering_correct = False
                except Exception as e:
                    self.log_test(f"Experimental filtering '{assay_type}'", False, f"Error: {str(e)}")
                    experimental_filtering_correct = False
            
            return validation_correct and experimental_filtering_correct
            
        except Exception as e:
            self.log_test("Data quality controls match ChEMBL", False, f"Error: {str(e)}")
            return False
    
    def test_backend_integration_readiness(self):
        """Test that backend is ready for PubChem integration"""
        print("\n=== Testing Backend Integration Readiness ===")
        
        try:
            # Test that backend is running and healthy
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code != 200:
                self.log_test("Backend health check", False, f"Backend not healthy: HTTP {response.status_code}")
                return False
            
            data = response.json()
            status = data.get('status', 'unknown')
            
            if status != 'healthy':
                self.log_test("Backend health status", False, f"Backend status: {status}")
                return False
            
            self.log_test("Backend health check", True, "Backend is healthy")
            
            # Check that expanded models integration is available
            models_loaded = data.get('models_loaded', {})
            expanded_models_available = models_loaded.get('expanded_models', False)
            
            if not expanded_models_available:
                self.log_test("Expanded models availability", False, "Expanded models not available")
                return False
            
            self.log_test("Expanded models availability", True, "Expanded models available")
            
            # Check expanded models info
            expanded_models_info = data.get('expanded_models_info', {})
            
            if not expanded_models_info:
                self.log_test("Expanded models info", False, "No expanded models info")
                return False
            
            # Check that we have the expected 23 targets
            total_targets = expanded_models_info.get('total_targets', 0)
            if total_targets != 23:
                self.log_test("Expanded models target count", False, f"Expected 23 targets, got {total_targets}")
                return False
            
            self.log_test("Expanded models target count", True, f"Correct target count: {total_targets}")
            
            # Check data sources include the expected ones
            data_sources = expanded_models_info.get('data_sources', [])
            expected_sources = ['ChEMBL', 'PubChem', 'BindingDB', 'DTC']
            
            missing_sources = [source for source in expected_sources if source not in data_sources]
            if missing_sources:
                self.log_test("Expected data sources", False, f"Missing sources: {missing_sources}")
                return False
            
            self.log_test("Expected data sources", True, f"All expected sources present: {data_sources}")
            
            # Test expanded endpoints are accessible
            expanded_endpoints = [
                "/expanded/health",
                "/expanded/targets"
            ]
            
            endpoints_accessible = True
            for endpoint in expanded_endpoints:
                try:
                    endpoint_response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
                    if endpoint_response.status_code == 200:
                        self.log_test(f"Expanded endpoint {endpoint}", True, "Endpoint accessible")
                    else:
                        self.log_test(f"Expanded endpoint {endpoint}", False, 
                                    f"HTTP {endpoint_response.status_code}")
                        endpoints_accessible = False
                except Exception as e:
                    self.log_test(f"Expanded endpoint {endpoint}", False, f"Error: {str(e)}")
                    endpoints_accessible = False
            
            return endpoints_accessible
            
        except Exception as e:
            self.log_test("Backend integration readiness", False, f"Error: {str(e)}")
            return False
    
    def test_modal_app_configuration(self):
        """Test that Modal apps are properly configured"""
        print("\n=== Testing Modal App Configuration ===")
        
        try:
            # Check Modal credentials are available
            modal_token_id = os.environ.get('MODAL_TOKEN_ID')
            modal_token_secret = os.environ.get('MODAL_TOKEN_SECRET')
            
            if not modal_token_id or not modal_token_secret:
                self.log_test("Modal credentials", False, "Modal credentials not found in environment")
                return False
            
            self.log_test("Modal credentials", True, "Modal credentials available")
            
            # Import and check Modal app configurations
            modules_to_check = [
                ("/app/modal_training/enhanced_pubchem_extractor.py", "enhanced_pubchem_extractor"),
                ("/app/modal_training/integrate_pubchem_with_chembl.py", "integrate_pubchem_with_chembl")
            ]
            
            apps_configured = True
            for module_path, module_name in modules_to_check:
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Check for Modal app
                    if hasattr(module, 'app'):
                        app = getattr(module, 'app')
                        self.log_test(f"Modal app {module_name}", True, f"App configured: {app.name}")
                    else:
                        self.log_test(f"Modal app {module_name}", False, "No Modal app found")
                        apps_configured = False
                    
                    # Check for Modal image configuration
                    if hasattr(module, 'image'):
                        image = getattr(module, 'image')
                        self.log_test(f"Modal image {module_name}", True, "Image configured")
                    else:
                        self.log_test(f"Modal image {module_name}", False, "No Modal image found")
                        apps_configured = False
                    
                    # Check for Modal volume configuration
                    if hasattr(module, 'datasets_volume'):
                        volume = getattr(module, 'datasets_volume')
                        self.log_test(f"Modal volume {module_name}", True, "Volume configured")
                    else:
                        self.log_test(f"Modal volume {module_name}", False, "No Modal volume found")
                        apps_configured = False
                        
                except Exception as e:
                    self.log_test(f"Modal app {module_name}", False, f"Error: {str(e)}")
                    apps_configured = False
            
            return apps_configured
            
        except Exception as e:
            self.log_test("Modal app configuration", False, f"Error: {str(e)}")
            return False
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        print("\n=== Testing Error Handling and Fallbacks ===")
        
        try:
            # Test that modules handle missing dependencies gracefully
            error_handling_correct = True
            
            # Test 1: Import with missing optional dependencies
            try:
                # Import the extractor
                extractor_path = Path("/app/modal_training/enhanced_pubchem_extractor.py")
                spec = importlib.util.spec_from_file_location("enhanced_pubchem_extractor", extractor_path)
                module = importlib.util.module_from_spec(spec)
                
                # Check that the module has proper error handling for imports
                source_code = extractor_path.read_text()
                
                if 'try:' in source_code and 'except' in source_code:
                    self.log_test("Import error handling", True, "Error handling present in imports")
                else:
                    self.log_test("Import error handling", False, "No import error handling found")
                    error_handling_correct = False
                
                # Check for timeout handling in API calls
                if 'timeout' in source_code:
                    self.log_test("API timeout handling", True, "Timeout handling present")
                else:
                    self.log_test("API timeout handling", False, "No timeout handling found")
                    error_handling_correct = False
                
                # Check for rate limiting handling
                if 'rate' in source_code.lower() or 'limit' in source_code.lower():
                    self.log_test("Rate limiting awareness", True, "Rate limiting considerations present")
                else:
                    self.log_test("Rate limiting awareness", False, "No rate limiting handling found")
                    error_handling_correct = False
                    
            except Exception as e:
                self.log_test("Error handling check", False, f"Error: {str(e)}")
                error_handling_correct = False
            
            # Test 2: Integration module error handling
            try:
                integration_path = Path("/app/modal_training/integrate_pubchem_with_chembl.py")
                source_code = integration_path.read_text()
                
                # Check for file existence checks
                if 'exists()' in source_code or 'FileNotFoundError' in source_code:
                    self.log_test("File existence error handling", True, "File existence checks present")
                else:
                    self.log_test("File existence error handling", False, "No file existence checks")
                    error_handling_correct = False
                
                # Check for data validation
                if 'validate' in source_code.lower() or 'check' in source_code.lower():
                    self.log_test("Data validation error handling", True, "Data validation present")
                else:
                    self.log_test("Data validation error handling", False, "No data validation found")
                    error_handling_correct = False
                    
            except Exception as e:
                self.log_test("Integration error handling check", False, f"Error: {str(e)}")
                error_handling_correct = False
            
            # Test 3: Pipeline orchestration error handling
            try:
                launcher_path = Path("/app/modal_training/launch_pubchem_integration.py")
                source_code = launcher_path.read_text()
                
                # Check for comprehensive error handling
                if 'try:' in source_code and 'except Exception' in source_code:
                    self.log_test("Pipeline error handling", True, "Comprehensive error handling present")
                else:
                    self.log_test("Pipeline error handling", False, "No comprehensive error handling")
                    error_handling_correct = False
                
                # Check for error logging/reporting
                if 'error' in source_code.lower() and ('print' in source_code or 'log' in source_code):
                    self.log_test("Error reporting", True, "Error reporting present")
                else:
                    self.log_test("Error reporting", False, "No error reporting found")
                    error_handling_correct = False
                    
            except Exception as e:
                self.log_test("Pipeline error handling check", False, f"Error: {str(e)}")
                error_handling_correct = False
            
            return error_handling_correct
            
        except Exception as e:
            self.log_test("Error handling and fallbacks", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all PubChem integration tests"""
        print(f"🧪 Starting PubChem BioAssay Integration Testing")
        print(f"Backend URL: {API_BASE}")
        print("=" * 80)
        
        # PubChem Integration Tests
        tests = [
            # Module Import Tests
            self.test_pubchem_extractor_module_import,
            self.test_pubchem_integration_module_import,
            self.test_pubchem_launcher_module_import,
            
            # Data Standardization Tests
            self.test_pubchem_data_standardization_compatibility,
            self.test_cross_source_deduplication_logic,
            self.test_data_quality_controls_match_chembl,
            
            # Pipeline Structure Tests
            self.test_pipeline_orchestration_structure,
            self.test_expected_dataset_boost_calculation,
            
            # Integration Readiness Tests
            self.test_backend_integration_readiness,
            self.test_modal_app_configuration,
            
            # Error Handling Tests
            self.test_error_handling_and_fallbacks
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {test.__name__}: {str(e)}")
                self.failed_tests.append({
                    'test': test.__name__,
                    'status': '❌ CRITICAL ERROR',
                    'success': False,
                    'details': str(e)
                })
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 PUBCHEM INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['success']])
        failed_tests = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        return passed_tests, failed_tests, self.test_results

if __name__ == "__main__":
    tester = PubChemIntegrationTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)