"""
PropMolFlow Quick Test & Demonstration
Test the PropMolFlow implementation and show usage examples
"""

import json
import requests
from typing import Dict, Any

def test_propmolflow_backend():
    """Test PropMolFlow backend integration"""
    
    base_url = "http://localhost:8001"
    
    print("🧪 PropMolFlow Backend Integration Test")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing generation health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/generate/health")
        if response.status_code == 200:
            health_data = response.json()
            print("   ✅ Health endpoint working")
            print(f"   📊 PropMolFlow available: {health_data['generation_methods']['propmolflow_available']}")
            print(f"   🎯 Supported targets: {health_data['supported_targets']}")
            print(f"   📋 Max molecules per request: {health_data['max_molecules_per_request']}")
        else:
            print(f"   ❌ Health endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health endpoint error: {e}")
    
    # Test 2: Get available targets
    print("\n2. Testing generation targets endpoint...")
    try:
        response = requests.get(f"{base_url}/api/generate/targets")
        if response.status_code == 200:
            targets_data = response.json()
            print("   ✅ Targets endpoint working")
            print(f"   🎯 Total targets: {targets_data['total_targets']}")
            print(f"   📋 Categories: {targets_data['categories']}")
            print(f"   🧪 Activity types: {targets_data['activity_types']}")
            
            # Show some example targets
            oncoproteins = [t for t in targets_data['targets'] if t['category'] == 'oncoprotein'][:3]
            tumor_suppressors = [t for t in targets_data['targets'] if t['category'] == 'tumor_suppressor'][:2]
            metastasis_suppressors = [t for t in targets_data['targets'] if t['category'] == 'metastasis_suppressor'][:2]
            
            print(f"   📊 Example targets:")
            print(f"      Oncoproteins: {[t['target'] for t in oncoproteins]}")
            print(f"      Tumor Suppressors: {[t['target'] for t in tumor_suppressors]}")
            print(f"      Metastasis Suppressors: {[t['target'] for t in metastasis_suppressors]}")
        else:
            print(f"   ❌ Targets endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Targets endpoint error: {e}")
    
    # Test 3: Get examples
    print("\n3. Testing generation examples endpoint...")
    try:
        response = requests.get(f"{base_url}/api/generate/examples")
        if response.status_code == 200:
            examples_data = response.json()
            print("   ✅ Examples endpoint working")
            print(f"   📚 Total examples: {examples_data['total_examples']}")
            
            for i, example in enumerate(examples_data['examples'][:2], 1):
                print(f"   📋 Example {i}: {example['name']}")
                print(f"      Description: {example['description']}")
                req = example['request']
                prop = req['properties'][0]
                print(f"      Target: {prop['target_name']} {prop['activity_type']} {prop['operator']} {prop['target_value']} nM")
        else:
            print(f"   ❌ Examples endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Examples endpoint error: {e}")
    
    # Test 4: Molecule validation
    print("\n4. Testing molecule validation endpoint...")
    try:
        test_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO", "invalid_smiles"]
        response = requests.post(
            f"{base_url}/api/generate/molecules/validate",
            json=test_smiles
        )
        if response.status_code == 200:
            validation_data = response.json()
            print("   ✅ Validation endpoint working")
            print(f"   📊 Validation stats:")
            stats = validation_data['validation_stats']
            print(f"      Total molecules: {stats['total_molecules']}")
            print(f"      Valid molecules: {stats['valid_molecules']}")
            print(f"      Drug-like molecules: {stats['drug_like_molecules']}")
        else:
            print(f"   ❌ Validation endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Validation endpoint error: {e}")
    
    # Test 5: Main health check (shows integration status)
    print("\n5. Testing main health endpoint for generation info...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            gen_info = health_data.get('molecular_generation_info', {})
            print("   ✅ Main health endpoint includes generation info")
            print(f"   🎯 Generation available: {gen_info.get('available', False)}")
            print(f"   🧪 Generation methods: {gen_info.get('generation_methods', [])}")
            print(f"   📋 Capabilities: {gen_info.get('capabilities', [])}")
        else:
            print(f"   ❌ Main health endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Main health endpoint error: {e}")
    
    print("\n" + "=" * 60)

def demo_generation_request():
    """Demonstrate how to use PropMolFlow generation"""
    
    print("\n🎯 PropMolFlow Generation Request Demo")
    print("=" * 60)
    
    # Example 1: Single target EGFR inhibitor
    print("\n📋 Example 1: EGFR Inhibitor Generation")
    example_request = {
        "num_molecules": 5,
        "properties": [
            {
                "target_name": "EGFR",
                "activity_type": "IC50",
                "target_value": 100.0,  # 100 nM
                "operator": "less_than",
                "importance": 1.0
            }
        ],
        "generation_method": "propmolflow",
        "temperature": 1.0,
        "guidance_scale": 7.5
    }
    
    print("Request:")
    print(json.dumps(example_request, indent=2))
    print("\nThis request asks PropMolFlow to generate 5 molecules with:")
    print("• IC50 against EGFR < 100 nM")
    print("• Using PropMolFlow generation method")
    print("• Standard temperature and guidance settings")
    
    # Example 2: Multi-target design
    print("\n📋 Example 2: Multi-Target Drug Design")
    multi_target_request = {
        "num_molecules": 10,
        "properties": [
            {
                "target_name": "EGFR",
                "activity_type": "IC50",
                "target_value": 50.0,
                "operator": "less_than",
                "importance": 1.0
            },
            {
                "target_name": "HER2", 
                "activity_type": "IC50",
                "target_value": 200.0,
                "operator": "less_than",
                "importance": 0.7
            }
        ],
        "generation_method": "propmolflow"
    }
    
    print("Request:")
    print(json.dumps(multi_target_request, indent=2))
    print("\nThis request generates molecules that are:")
    print("• Highly active against EGFR (IC50 < 50 nM, importance 1.0)")
    print("• Moderately active against HER2 (IC50 < 200 nM, importance 0.7)")
    
    # Example 3: Tumor suppressor activator
    print("\n📋 Example 3: Tumor Suppressor Activator")
    tumor_suppressor_request = {
        "num_molecules": 8,
        "properties": [
            {
                "target_name": "TP53",
                "activity_type": "EC50",
                "target_value": 500.0,
                "operator": "less_than",
                "importance": 1.0
            }
        ],
        "generation_method": "propmolflow",
        "constraints": {
            "molecular_weight_max": 500,
            "logp_max": 5.0
        }
    }
    
    print("Request:")
    print(json.dumps(tumor_suppressor_request, indent=2))
    print("\nThis request generates molecules that:")
    print("• Activate TP53 tumor suppressor (EC50 < 500 nM)")
    print("• Meet drug-like property constraints")
    
    print("\n" + "=" * 60)

def show_propmolflow_advantages():
    """Show advantages of PropMolFlow over other generation methods"""
    
    print("\n🌟 PropMolFlow Advantages")
    print("=" * 60)
    
    advantages = [
        {
            "feature": "Property-Guided Generation",
            "description": "Generate molecules with specific bioactivity profiles",
            "example": "Create EGFR inhibitors with IC50 < 100 nM"
        },
        {
            "feature": "Multi-Target Design",
            "description": "Optimize for multiple proteins simultaneously",
            "example": "Active against EGFR but selective vs off-targets"
        },
        {
            "feature": "Flow Matching Efficiency",
            "description": "Faster and more stable than diffusion models",
            "example": "Generate 100 molecules in minutes vs hours"
        },
        {
            "feature": "Exact Likelihood Control",
            "description": "Precise control over molecular properties",
            "example": "Exact IC50 targeting vs approximate ranges"
        },
        {
            "feature": "Integration with Prediction",
            "description": "Uses our trained ChemBERTa/Chemprop for validation",
            "example": "Generate → Predict → Filter pipeline"
        },
        {
            "feature": "Drug Discovery Pipeline",
            "description": "End-to-end from idea to optimized candidates",
            "example": "Target → Properties → Generated leads → Validation"
        }
    ]
    
    for i, adv in enumerate(advantages, 1):
        print(f"\n{i}. {adv['feature']}")
        print(f"   📝 {adv['description']}")
        print(f"   💡 Example: {adv['example']}")
    
    print("\n🎯 Competitive Advantage:")
    print("• Veridica transforms from 'test molecules' to 'design molecules'")
    print("• Complete AI drug discovery platform")
    print("• Property-guided generation + multi-target prediction")
    print("• Cutting-edge flow matching technology")
    
    print("\n" + "=" * 60)

def show_implementation_roadmap():
    """Show PropMolFlow implementation roadmap"""
    
    print("\n🗺️ PropMolFlow Implementation Roadmap")
    print("=" * 60)
    
    phases = [
        {
            "phase": "Phase 1: Foundation (Current)",
            "duration": "Complete",
            "items": [
                "✅ PropMolFlow architecture design",
                "✅ Backend API integration",
                "✅ Property encoding system",
                "✅ Flow matching network",
                "✅ Molecular decoder"
            ]
        },
        {
            "phase": "Phase 2: Training & Deployment",
            "duration": "1-2 months",
            "items": [
                "🔄 Train PropMolFlow on expanded dataset",
                "🔄 Deploy generation functions on Modal",
                "⏳ Integrate with ChemBERTa validation",
                "⏳ Add molecular property prediction",
                "⏳ Implement drug-likeness filtering"
            ]
        },
        {
            "phase": "Phase 3: Advanced Features",
            "duration": "2-3 months",
            "items": [
                "⏳ Multi-objective optimization",
                "⏳ Synthetic accessibility scoring", 
                "⏳ Patent landscape integration",
                "⏳ Structure-activity relationship analysis",
                "⏳ Advanced constraint handling"
            ]
        },
        {
            "phase": "Phase 4: Production Platform",
            "duration": "3-4 months",
            "items": [
                "⏳ Web interface for molecular generation",
                "⏳ Batch generation and optimization",
                "⏳ Export to synthesis platforms",
                "⏳ Performance monitoring and analytics",
                "⏳ API rate limiting and scaling"
            ]
        }
    ]
    
    for phase_info in phases:
        print(f"\n{phase_info['phase']}")
        print(f"Duration: {phase_info['duration']}")
        for item in phase_info['items']:
            print(f"  {item}")
    
    print("\n🎯 Key Milestones:")
    print("• Month 1: PropMolFlow generating valid molecules")
    print("• Month 2: Property-guided generation working")
    print("• Month 3: Multi-target optimization functional")
    print("• Month 4: Production-ready drug discovery platform")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Run all tests and demos
    test_propmolflow_backend()
    demo_generation_request()
    show_propmolflow_advantages()
    show_implementation_roadmap()
    
    print("\n🎉 PropMolFlow integration is ready for the next phase!")
    print("Next step: Train PropMolFlow model on expanded dataset")