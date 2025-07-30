"""
Run the training fix tests on Modal
"""

import modal
from test_training_minimal import app

if __name__ == "__main__":
    print("🚀 Testing ChemBERTa and Chemprop fixes on Modal...")
    
    with app.run():
        print("\n🧪 Testing ChemBERTa components...")
        chemberta_result = app.test_chemberta_minimal.remote()
        print(f"ChemBERTa result: {chemberta_result}")
        
        print("\n🧪 Testing Chemprop components...")
        chemprop_result = app.test_chemprop_minimal.remote()
        print(f"Chemprop result: {chemprop_result}")
        
        if (chemberta_result.get('status') == 'success' and 
            chemprop_result.get('status') == 'success'):
            print("\n✅ All training component fixes validated successfully!")
            print("🎯 Ready to run full training with bug fixes")
        else:
            print("\n❌ Some issues remain - check logs above")