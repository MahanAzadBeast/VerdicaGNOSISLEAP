#!/usr/bin/env python3
"""
Execute End-to-End Training Pipeline Tests
Validates ChemBERTa and Chemprop fixes on Modal
"""

import modal
from test_end_to_end_training import test_app

def main():
    print("🚀 EXECUTING END-TO-END TRAINING PIPELINE TESTS")
    print("=" * 60)
    print("Testing:")
    print("1. ChemBERTa training with device property bug fix")
    print("2. Chemprop training with CLI compatibility fix") 
    print("3. W&B logging integration")
    print("=" * 60)
    
    try:
        with test_app.run():
            print("\n📊 Step 1: Creating test dataset...")
            dataset_result = test_app.create_test_dataset.remote()
            print(f"Dataset creation result: {dataset_result}")
            
            if dataset_result.get('status') != 'success':
                print("❌ Dataset creation failed - cannot proceed with training tests")
                return
            
            print("\n🧬 Step 2: Testing ChemBERTa training (Device Bug Fix)...")
            try:
                chemberta_result = test_app.test_chemberta_training.remote()
                print(f"ChemBERTa result: {chemberta_result}")
            except Exception as e:
                print(f"❌ ChemBERTa test failed with exception: {e}")
                chemberta_result = {"status": "error", "message": f"Exception: {str(e)}"}
            
            print("\n⚗️ Step 3: Testing Chemprop training (CLI Compatibility Fix)...")
            try:
                chemprop_result = test_app.test_chemprop_training.remote()
                print(f"Chemprop result: {chemprop_result}")
            except Exception as e:
                print(f"❌ Chemprop test failed with exception: {e}")
                chemprop_result = {"status": "error", "message": f"Exception: {str(e)}"}
            
            print("\n📋 Step 4: Summarizing results...")
            summary = test_app.summarize_results.remote(dataset_result, chemberta_result, chemprop_result)
            print(f"Final summary: {summary}")
            
            return summary
            
    except Exception as e:
        print(f"❌ Critical error in test execution: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    result = main()
    
    if result and result.get('all_passed', False):
        print("\n🎉 SUCCESS: All training pipeline fixes validated!")
        exit(0)
    else:
        print("\n⚠️ WARNING: Some issues found - check logs above")
        exit(1)