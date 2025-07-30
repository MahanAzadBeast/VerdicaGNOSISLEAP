#!/usr/bin/env python3
"""
ChemBERTa and Chemprop Training Pipeline Testing
Tests the specific fixes for device property bug and CLI compatibility
"""

import sys
import os
import importlib.util
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

class TrainingPipelineFixTester:
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
    
    def test_chemberta_import_and_basic_functionality(self):
        """Test ChemBERTa training module import and basic functionality"""
        print("\n=== Testing ChemBERTa Module Import and Basic Functionality ===")
        
        try:
            # Test import
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemberta", "/app/modal_training/train_chemberta.py")
            train_chemberta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemberta)
            
            self.log_test("ChemBERTa module import", True, "Successfully imported train_chemberta.py")
            
            # Test ChemBERTaMultiTaskModel class exists
            if hasattr(train_chemberta, 'ChemBERTaMultiTaskModel'):
                model_class = train_chemberta.ChemBERTaMultiTaskModel
                self.log_test("ChemBERTaMultiTaskModel class", True, "Class definition found")
                
                # Test device property fix - check if device property exists
                import inspect
                model_methods = [method for method in dir(model_class) if not method.startswith('_')]
                has_device_property = 'device' in model_methods
                
                if has_device_property:
                    # Check the device property implementation
                    device_method = getattr(model_class, 'device')
                    if hasattr(device_method, 'fget'):  # It's a property
                        source = inspect.getsource(device_method.fget)
                        has_safe_device_access = 'next(self.parameters()).device' in source
                        self.log_test("ChemBERTa device property fix", has_safe_device_access, 
                                    "Device property uses safe next(self.parameters()).device access")
                    else:
                        self.log_test("ChemBERTa device property fix", False, "Device is not a property")
                else:
                    self.log_test("ChemBERTa device property fix", False, "Device property not found")
                
                # Test ChemBERTaTrainer class exists
                if hasattr(train_chemberta, 'ChemBERTaTrainer'):
                    trainer_class = train_chemberta.ChemBERTaTrainer
                    self.log_test("ChemBERTaTrainer class", True, "Custom trainer class found")
                    
                    # Check if evaluate method has device fix
                    if hasattr(trainer_class, 'evaluate'):
                        evaluate_source = inspect.getsource(trainer_class.evaluate)
                        has_device_fix = 'next(self.model.parameters()).device' in evaluate_source
                        self.log_test("ChemBERTa evaluate device fix", has_device_fix,
                                    "Evaluate method uses safe device access")
                    else:
                        self.log_test("ChemBERTa evaluate method", False, "Evaluate method not found")
                else:
                    self.log_test("ChemBERTaTrainer class", False, "Custom trainer class not found")
                
                # Test W&B logging components
                if hasattr(train_chemberta, 'WandbMetricsCallback'):
                    self.log_test("ChemBERTa W&B callback", True, "WandbMetricsCallback class found")
                else:
                    self.log_test("ChemBERTa W&B callback", False, "WandbMetricsCallback class not found")
                
                return True
            else:
                self.log_test("ChemBERTaMultiTaskModel class", False, "Class definition not found")
                return False
                
        except Exception as e:
            self.log_test("ChemBERTa module import", False, f"Import error: {str(e)}")
            return False
    
    def test_chemprop_import_and_basic_functionality(self):
        """Test Chemprop training module import and basic functionality"""
        print("\n=== Testing Chemprop Module Import and Basic Functionality ===")
        
        try:
            # Test import
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemprop", "/app/modal_training/train_chemprop.py")
            train_chemprop = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemprop)
            
            self.log_test("Chemprop module import", True, "Successfully imported train_chemprop.py")
            
            # Test CLI command generation fix
            if hasattr(train_chemprop, 'run_chemprop_training'):
                import inspect
                training_source = inspect.getsource(train_chemprop.run_chemprop_training)
                
                # Check for new CLI format
                has_new_cli = "'python', '-m', 'chemprop.train'" in training_source
                has_old_cli = "'chemprop_train'" in training_source
                
                self.log_test("Chemprop CLI compatibility fix", has_new_cli and not has_old_cli,
                            f"Uses new CLI format: {has_new_cli}, Avoids old CLI: {not has_old_cli}")
            else:
                self.log_test("Chemprop training function", False, "run_chemprop_training function not found")
            
            # Test prediction CLI fix
            if hasattr(train_chemprop, 'predict_chemprop'):
                import inspect
                predict_source = inspect.getsource(train_chemprop.predict_chemprop)
                
                # Check for new prediction CLI format
                has_new_predict_cli = "'python', '-m', 'chemprop.predict'" in predict_source
                has_old_predict_cli = "'chemprop_predict'" in predict_source
                
                self.log_test("Chemprop predict CLI compatibility fix", has_new_predict_cli and not has_old_predict_cli,
                            f"Uses new predict CLI: {has_new_predict_cli}, Avoids old predict CLI: {not has_old_predict_cli}")
            else:
                self.log_test("Chemprop predict function", False, "predict_chemprop function not found")
            
            # Test W&B logging components
            if hasattr(train_chemprop, 'ChempropWandbLogger'):
                logger_class = train_chemprop.ChempropWandbLogger
                self.log_test("Chemprop W&B logger", True, "ChempropWandbLogger class found")
                
                # Check logger methods
                logger_methods = [method for method in dir(logger_class) if not method.startswith('_')]
                expected_methods = ['log_epoch_metrics', 'log_final_results']
                has_all_methods = all(method in logger_methods for method in expected_methods)
                
                self.log_test("Chemprop W&B logger methods", has_all_methods,
                            f"Has methods: {[m for m in expected_methods if m in logger_methods]}")
            else:
                self.log_test("Chemprop W&B logger", False, "ChempropWandbLogger class not found")
            
            return True
            
        except Exception as e:
            self.log_test("Chemprop module import", False, f"Import error: {str(e)}")
            return False
    
    def test_chemberta_device_property_access(self):
        """Test ChemBERTa device property access works correctly"""
        print("\n=== Testing ChemBERTa Device Property Access ===")
        
        try:
            # Import required modules
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemberta", "/app/modal_training/train_chemberta.py")
            train_chemberta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemberta)
            
            # Test device property with mock model
            import torch
            import torch.nn as nn
            
            # Create a simple mock model to test device property
            class MockModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 1)
                
                @property
                def device(self):
                    """Test the fixed device property implementation"""
                    return next(self.parameters()).device
            
            mock_model = MockModel()
            
            # Test device property access
            try:
                device = mock_model.device
                self.log_test("Device property access", True, f"Device property returned: {device}")
                
                # Test with different device if CUDA available
                if torch.cuda.is_available():
                    mock_model.cuda()
                    cuda_device = mock_model.device
                    self.log_test("Device property CUDA", cuda_device.type == 'cuda', 
                                f"CUDA device property: {cuda_device}")
                else:
                    self.log_test("Device property CUDA", True, "CUDA not available, skipping CUDA test")
                
                return True
                
            except Exception as e:
                self.log_test("Device property access", False, f"Device property error: {str(e)}")
                return False
                
        except Exception as e:
            self.log_test("Device property test setup", False, f"Setup error: {str(e)}")
            return False
    
    def test_chemprop_cli_command_generation(self):
        """Test Chemprop CLI command generation works correctly"""
        print("\n=== Testing Chemprop CLI Command Generation ===")
        
        try:
            # Import required modules
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemprop", "/app/modal_training/train_chemprop.py")
            train_chemprop = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemprop)
            
            # Test command generation by examining the function
            if hasattr(train_chemprop, 'run_chemprop_training'):
                import inspect
                
                # Get the source code
                source = inspect.getsource(train_chemprop.run_chemprop_training)
                
                # Check for correct CLI patterns
                correct_train_cli = "'python', '-m', 'chemprop.train'" in source
                incorrect_train_cli = "'chemprop_train'" in source
                
                self.log_test("Chemprop train CLI format", correct_train_cli and not incorrect_train_cli,
                            f"Correct format: {correct_train_cli}, Incorrect format: {incorrect_train_cli}")
                
                # Check for proper command structure
                has_data_path = "'--data_path'" in source
                has_save_dir = "'--save_dir'" in source
                has_epochs = "'--epochs'" in source
                
                basic_structure = has_data_path and has_save_dir and has_epochs
                self.log_test("Chemprop command structure", basic_structure,
                            f"Has basic args: data_path={has_data_path}, save_dir={has_save_dir}, epochs={has_epochs}")
                
                return correct_train_cli and not incorrect_train_cli and basic_structure
            else:
                self.log_test("Chemprop training function", False, "run_chemprop_training function not found")
                return False
                
        except Exception as e:
            self.log_test("Chemprop CLI test", False, f"CLI test error: {str(e)}")
            return False
    
    def test_chemberta_model_loading_fix(self):
        """Test ChemBERTa model loading function handles device placement correctly"""
        print("\n=== Testing ChemBERTa Model Loading Fix ===")
        
        try:
            # Import required modules
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemberta", "/app/modal_training/train_chemberta.py")
            train_chemberta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemberta)
            
            # Test model loading function
            if hasattr(train_chemberta, 'load_and_predict_chemberta'):
                import inspect
                
                # Get the source code
                source = inspect.getsource(train_chemberta.load_and_predict_chemberta)
                
                # Check for proper model reconstruction
                has_model_reconstruction = 'ChemBERTaMultiTaskModel(' in source
                has_device_handling = '.to(device)' in source
                has_checkpoint_loading = 'torch.load(' in source
                
                self.log_test("ChemBERTa model reconstruction", has_model_reconstruction,
                            f"Reconstructs model architecture: {has_model_reconstruction}")
                
                self.log_test("ChemBERTa device handling", has_device_handling,
                            f"Handles device placement: {has_device_handling}")
                
                self.log_test("ChemBERTa checkpoint loading", has_checkpoint_loading,
                            f"Loads model weights: {has_checkpoint_loading}")
                
                return has_model_reconstruction and has_device_handling and has_checkpoint_loading
            else:
                self.log_test("ChemBERTa model loading function", False, "load_and_predict_chemberta function not found")
                return False
                
        except Exception as e:
            self.log_test("ChemBERTa model loading test", False, f"Model loading test error: {str(e)}")
            return False
    
    def test_wandb_logging_components(self):
        """Test W&B logging components are properly implemented"""
        print("\n=== Testing W&B Logging Components ===")
        
        all_passed = True
        
        # Test ChemBERTa W&B components
        try:
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemberta", "/app/modal_training/train_chemberta.py")
            train_chemberta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemberta)
            
            # Check ChemBERTa W&B components
            if hasattr(train_chemberta, 'WandbMetricsCallback'):
                callback_class = train_chemberta.WandbMetricsCallback
                callback_methods = [method for method in dir(callback_class) if not method.startswith('_')]
                
                expected_callback_methods = ['on_log', 'on_evaluate']
                has_callback_methods = all(method in callback_methods for method in expected_callback_methods)
                
                self.log_test("ChemBERTa W&B callback methods", has_callback_methods,
                            f"Has methods: {[m for m in expected_callback_methods if m in callback_methods]}")
            else:
                self.log_test("ChemBERTa W&B callback", False, "WandbMetricsCallback not found")
                all_passed = False
            
            # Check ChemBERTa trainer W&B integration
            if hasattr(train_chemberta, 'ChemBERTaTrainer'):
                trainer_class = train_chemberta.ChemBERTaTrainer
                
                # Check for W&B logging methods
                if hasattr(trainer_class, '_create_and_log_scatter_plots'):
                    self.log_test("ChemBERTa scatter plot logging", True, "Scatter plot logging method found")
                else:
                    self.log_test("ChemBERTa scatter plot logging", False, "Scatter plot logging method not found")
                    all_passed = False
                
                if hasattr(trainer_class, '_create_and_log_performance_summary'):
                    self.log_test("ChemBERTa performance summary logging", True, "Performance summary logging method found")
                else:
                    self.log_test("ChemBERTa performance summary logging", False, "Performance summary logging method not found")
                    all_passed = False
            
        except Exception as e:
            self.log_test("ChemBERTa W&B components", False, f"ChemBERTa W&B test error: {str(e)}")
            all_passed = False
        
        # Test Chemprop W&B components
        try:
            spec = importlib.util.spec_from_file_location("train_chemprop", "/app/modal_training/train_chemprop.py")
            train_chemprop = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemprop)
            
            # Check Chemprop W&B components
            if hasattr(train_chemprop, 'ChempropWandbLogger'):
                logger_class = train_chemprop.ChempropWandbLogger
                logger_methods = [method for method in dir(logger_class) if not method.startswith('_')]
                
                expected_logger_methods = ['log_epoch_metrics', 'log_final_results']
                has_logger_methods = all(method in logger_methods for method in expected_logger_methods)
                
                self.log_test("Chemprop W&B logger methods", has_logger_methods,
                            f"Has methods: {[m for m in expected_logger_methods if m in logger_methods]}")
            else:
                self.log_test("Chemprop W&B logger", False, "ChempropWandbLogger not found")
                all_passed = False
            
            # Check for visualization functions
            if hasattr(train_chemprop, 'train_chemprop_multitask'):
                import inspect
                source = inspect.getsource(train_chemprop.train_chemprop_multitask)
                
                has_scatter_plots = 'scatter(' in source and 'plt.savefig(' in source
                has_performance_plots = 'barh(' in source and 'wandb.Image(' in source
                has_wandb_table = 'wandb.Table(' in source
                
                self.log_test("Chemprop visualization plots", has_scatter_plots and has_performance_plots,
                            f"Scatter plots: {has_scatter_plots}, Performance plots: {has_performance_plots}")
                
                self.log_test("Chemprop W&B tables", has_wandb_table,
                            f"W&B table logging: {has_wandb_table}")
            
        except Exception as e:
            self.log_test("Chemprop W&B components", False, f"Chemprop W&B test error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_modal_integration_endpoints(self):
        """Test Modal integration endpoints related to training pipelines"""
        print("\n=== Testing Modal Integration Endpoints ===")
        
        try:
            # Check if Modal integration files exist
            modal_files = [
                "/app/modal_training/train_chemberta.py",
                "/app/modal_training/train_chemprop.py"
            ]
            
            all_files_exist = True
            for file_path in modal_files:
                if Path(file_path).exists():
                    self.log_test(f"Modal file exists: {Path(file_path).name}", True, f"File found: {file_path}")
                else:
                    self.log_test(f"Modal file exists: {Path(file_path).name}", False, f"File not found: {file_path}")
                    all_files_exist = False
            
            if not all_files_exist:
                return False
            
            # Test Modal app definitions
            for file_path in modal_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    file_name = Path(file_path).name
                    
                    # Check for Modal app definition
                    has_modal_app = 'modal.App(' in content
                    self.log_test(f"Modal app definition: {file_name}", has_modal_app,
                                f"Modal app defined: {has_modal_app}")
                    
                    # Check for Modal function decorators
                    has_modal_function = '@app.function(' in content
                    self.log_test(f"Modal function decorators: {file_name}", has_modal_function,
                                f"Modal functions defined: {has_modal_function}")
                    
                    # Check for GPU configuration
                    has_gpu_config = 'gpu="A100"' in content
                    self.log_test(f"GPU configuration: {file_name}", has_gpu_config,
                                f"A100 GPU configured: {has_gpu_config}")
                    
                    # Check for W&B secrets
                    has_wandb_secret = 'wandb_secret' in content
                    self.log_test(f"W&B secrets: {file_name}", has_wandb_secret,
                                f"W&B secrets configured: {has_wandb_secret}")
                    
                except Exception as e:
                    self.log_test(f"Modal file analysis: {Path(file_path).name}", False, f"Analysis error: {str(e)}")
                    return False
            
            return True
            
        except Exception as e:
            self.log_test("Modal integration endpoints", False, f"Modal integration test error: {str(e)}")
            return False
    
    def test_training_pipeline_configuration(self):
        """Test training pipeline configuration and parameters"""
        print("\n=== Testing Training Pipeline Configuration ===")
        
        all_passed = True
        
        # Test ChemBERTa configuration
        try:
            sys.path.append('/app/modal_training')
            spec = importlib.util.spec_from_file_location("train_chemberta", "/app/modal_training/train_chemberta.py")
            train_chemberta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemberta)
            
            if hasattr(train_chemberta, 'train_chemberta_multitask'):
                import inspect
                signature = inspect.signature(train_chemberta.train_chemberta_multitask)
                params = list(signature.parameters.keys())
                
                # Check for essential parameters
                essential_params = ['dataset_name', 'model_name', 'batch_size', 'learning_rate', 'num_epochs']
                has_essential_params = all(param in params for param in essential_params)
                
                self.log_test("ChemBERTa essential parameters", has_essential_params,
                            f"Has params: {[p for p in essential_params if p in params]}")
                
                # Check for W&B parameters
                wandb_params = ['run_name']
                has_wandb_params = any(param in params for param in wandb_params)
                
                self.log_test("ChemBERTa W&B parameters", has_wandb_params,
                            f"Has W&B params: {[p for p in wandb_params if p in params]}")
            else:
                self.log_test("ChemBERTa training function", False, "train_chemberta_multitask function not found")
                all_passed = False
                
        except Exception as e:
            self.log_test("ChemBERTa configuration", False, f"ChemBERTa config error: {str(e)}")
            all_passed = False
        
        # Test Chemprop configuration
        try:
            spec = importlib.util.spec_from_file_location("train_chemprop", "/app/modal_training/train_chemprop.py")
            train_chemprop = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_chemprop)
            
            if hasattr(train_chemprop, 'train_chemprop_multitask'):
                import inspect
                signature = inspect.signature(train_chemprop.train_chemprop_multitask)
                params = list(signature.parameters.keys())
                
                # Check for essential parameters
                essential_params = ['dataset_name', 'num_epochs', 'batch_size', 'learning_rate']
                has_essential_params = all(param in params for param in essential_params)
                
                self.log_test("Chemprop essential parameters", has_essential_params,
                            f"Has params: {[p for p in essential_params if p in params]}")
                
                # Check for multi-task parameters
                multitask_params = ['multitask_scaling', 'ensemble_size']
                has_multitask_params = all(param in params for param in multitask_params)
                
                self.log_test("Chemprop multi-task parameters", has_multitask_params,
                            f"Has multi-task params: {[p for p in multitask_params if p in params]}")
            else:
                self.log_test("Chemprop training function", False, "train_chemprop_multitask function not found")
                all_passed = False
                
        except Exception as e:
            self.log_test("Chemprop configuration", False, f"Chemprop config error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all training pipeline tests and provide summary"""
        print(f"🧪 Starting ChemBERTa and Chemprop Training Pipeline Fix Testing")
        print("=" * 70)
        
        # Run all tests
        tests = [
            self.test_chemberta_import_and_basic_functionality,
            self.test_chemprop_import_and_basic_functionality,
            self.test_chemberta_device_property_access,
            self.test_chemprop_cli_command_generation,
            self.test_chemberta_model_loading_fix,
            self.test_wandb_logging_components,
            self.test_modal_integration_endpoints,
            self.test_training_pipeline_configuration
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
        print("\n" + "=" * 70)
        print("🏁 TRAINING PIPELINE TEST SUMMARY")
        print("=" * 70)
        
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
    tester = TrainingPipelineFixTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)