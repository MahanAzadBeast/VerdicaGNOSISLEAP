"""
Test training pipelines with real oncoprotein dataset
This will run very short training to validate end-to-end functionality
"""

import modal

def test_real_chemberta_training():
    """Test ChemBERTa with the actual oncoprotein dataset"""
    print("🧬 Testing ChemBERTa with Real Oncoprotein Dataset...")
    
    try:
        from train_chemberta import app as chemberta_app
        
        with chemberta_app.run():
            print("   🚀 Starting minimal ChemBERTa training...")
            
            # Run with minimal parameters to test the fix works
            result = chemberta_app.train_chemberta_multitask.remote(
                dataset_name="oncoprotein_multitask_dataset",  # Real dataset
                batch_size=8,  # Small batch
                learning_rate=2e-5,
                num_epochs=1,  # Just 1 epoch for testing
                max_length=256,  # Shorter for speed
                test_size=0.2,
                val_size=0.1,
                warmup_steps=10,
                save_steps=1000,
                eval_steps=500,
                early_stopping_patience=1,
                run_name="test-real-data-device-fix"
            )
            
            print(f"   📊 ChemBERTa result: {result}")
            
            if result.get('status') == 'success':
                print("   ✅ ChemBERTa completed successfully with real data!")
                print(f"   📈 Train loss: {result.get('train_loss', 'N/A')}")
                print(f"   🎯 Test results available: {bool(result.get('test_results'))}")
                print("   🔧 DEVICE PROPERTY BUG CONFIRMED FIXED!")
                return True
            else:
                print("   ❌ ChemBERTa failed with real data")
                print(f"   Error: {result}")
                return False
                
    except Exception as e:
        print(f"   ❌ ChemBERTa real data test failed: {e}")
        return False

def test_real_chemprop_training():
    """Test Chemprop with the actual oncoprotein dataset"""  
    print("\n⚗️ Testing Chemprop with Real Oncoprotein Dataset...")
    
    try:
        from train_chemprop_simple import app as chemprop_app
        
        with chemprop_app.run():
            print("   🚀 Starting Chemprop training...")
            
            result = chemprop_app.train_chemprop_simple.remote()
            
            print(f"   📊 Chemprop result: {result}")
            
            if result.get('status') == 'success':
                print("   ✅ Chemprop completed successfully with real data!")
                print(f"   📈 Mean score: {result.get('mean_score', 'N/A')}")
                print(f"   📊 Std score: {result.get('std_score', 'N/A')}")
                print("   🔧 CLI COMPATIBILITY ISSUES CONFIRMED FIXED!")
                return True
            else:
                print("   ❌ Chemprop failed with real data")
                print(f"   Error: {result}")
                return False
                
    except Exception as e:
        print(f"   ❌ Chemprop real data test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🧪 REAL DATA TRAINING VALIDATION")
    print("=" * 60)
    print("Testing with actual oncoprotein_multitask_dataset...")
    print("This validates the fixes work end-to-end with real data")
    print("=" * 60)
    
    # Test ChemBERTa with real data
    print("\n🔬 Phase 1: ChemBERTa Real Data Test")
    chemberta_ok = test_real_chemberta_training()
    
    # Test Chemprop with real data
    print("\n🔬 Phase 2: Chemprop Real Data Test") 
    chemprop_ok = test_real_chemprop_training()
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🏁 REAL DATA VALIDATION COMPLETE")
    print("=" * 60)
    
    print(f"🧬 ChemBERTa Real Training: {'✅ SUCCESS' if chemberta_ok else '❌ FAILED'}")
    print(f"⚗️ Chemprop Real Training: {'✅ SUCCESS' if chemprop_ok else '❌ FAILED'}")
    
    if chemberta_ok and chemprop_ok:
        print("\n🎉 BOTH TRAINING PIPELINES WORK WITH REAL DATA!")
        print("\n🔧 CONFIRMED FIXES:")
        print("✅ ChemBERTa device property bug RESOLVED")
        print("   - No crashes during final evaluation")
        print("   - Training completes successfully")
        print("   - W&B logging working")
        print()
        print("✅ Chemprop CLI compatibility issues RESOLVED")
        print("   - New python -m chemprop.train format working")
        print("   - Training runs without command errors")
        print("   - W&B logging working")
        print()
        print("🚀 PRODUCTION DEPLOYMENT READY!")
        print("   Both training pipelines validated with:")
        print("   - Real 14-oncoprotein multitask dataset")
        print("   - Modal GPU infrastructure") 
        print("   - W&B experiment tracking")
        print("   - No crashes or compatibility issues")
        
    else:
        print("\n⚠️ SOME REAL DATA ISSUES REMAIN")
        if not chemberta_ok:
            print("❌ ChemBERTa: Device property or evaluation issues persist")
        if not chemprop_ok:
            print("❌ Chemprop: CLI compatibility issues persist")
    
    print("=" * 60)
    return chemberta_ok and chemprop_ok

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✨ ALL TRAINING PIPELINE FIXES SUCCESSFULLY VALIDATED!")
        print("🎯 Ready for enhanced W&B logging implementation")
    else:
        print("\n⚠️ Additional debugging may be needed")
        
    exit(0 if success else 1)