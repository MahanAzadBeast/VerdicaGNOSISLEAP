#!/usr/bin/env python3
"""
AI Agent Model Registry Demo
Demonstrates how an AI agent would discover and use models from the registry
"""
import asyncio
import requests
import json
from typing import Dict, List, Optional

class AIAgent:
    """Demo AI Agent that uses the Model Registry"""
    
    def __init__(self, registry_url: str = "http://localhost:8001/api/registry"):
        self.registry_url = registry_url
        self.name = "GNOSIS AI Agent"
    
    def discover_capabilities(self) -> Dict:
        """Discover what models and capabilities are available"""
        print(f"🤖 {self.name}: Discovering available capabilities...")
        
        response = requests.get(f"{self.registry_url}/discover/summary")
        if response.status_code != 200:
            print(f"❌ Failed to discover capabilities: {response.status_code}")
            return {}
        
        summary = response.json()
        print(f"   Found {summary['total_models']} models in {len(summary['categories'])} categories")
        
        for category, count in summary['categories'].items():
            print(f"   📁 {category}: {count} models")
        
        return summary
    
    def find_best_model_for_task(self, task: str) -> Optional[Dict]:
        """Find the best model for a specific task"""
        print(f"🔍 {self.name}: Looking for best model for '{task}' task...")
        
        response = requests.get(f"{self.registry_url}/discover/best", 
                              params={"task_type": task})
        
        if response.status_code != 200:
            print(f"   ❌ No models found for task '{task}'")
            return None
        
        result = response.json()
        best_model = result['best_model']
        performance = result['model_info']['capabilities']['performance']
        
        print(f"   ✅ Best model: {best_model}")
        print(f"   📊 Performance: {performance}")
        
        return result
    
    def get_model_info(self, model_slug: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        print(f"📋 {self.name}: Getting info for model '{model_slug}'...")
        
        response = requests.get(f"{self.registry_url}/models/{model_slug}/latest")
        
        if response.status_code != 200:
            print(f"   ❌ Model '{model_slug}' not found")
            return None
        
        model_info = response.json()
        
        print(f"   📝 Model: {model_info['model_slug']} v{model_info['semver']}")
        print(f"   🏷️  Stage: {model_info['stage']}")
        print(f"   📦 Artifacts: {len(model_info['artifacts'])} files")
        print(f"   📊 Metrics: {len(model_info['metrics'])} metrics")
        
        return model_info
    
    def plan_drug_discovery_workflow(self, compound_smiles: str) -> Dict:
        """Plan a complete drug discovery workflow using available models"""
        print(f"\n🧬 {self.name}: Planning drug discovery workflow for compound:")
        print(f"   SMILES: {compound_smiles}")
        
        # Discover available capabilities
        capabilities = self.discover_capabilities()
        
        workflow = {
            "compound": compound_smiles,
            "steps": [],
            "models_used": []
        }
        
        # Step 1: Ligand Activity Prediction
        if "ligand-activity" in capabilities.get("categories", {}):
            ligand_model = self.find_best_model_for_task("ligand-activity")
            if ligand_model:
                workflow["steps"].append({
                    "step": 1,
                    "task": "Ligand Activity Prediction",
                    "model": ligand_model["best_model"],
                    "purpose": "Predict IC50/Ki/EC50 for oncology targets",
                    "output": "Target binding affinities"
                })
                workflow["models_used"].append(ligand_model["best_model"])
        
        # Step 2: Cytotoxicity Prediction
        if "cytotoxicity" in capabilities.get("categories", {}):
            cytotox_model = self.find_best_model_for_task("cytotoxicity")
            if cytotox_model:
                workflow["steps"].append({
                    "step": 2,
                    "task": "Cytotoxicity Prediction",
                    "model": cytotox_model["best_model"],
                    "purpose": "Predict IC50 for cancer cell lines",
                    "output": "Cell line sensitivity profiles"
                })
                workflow["models_used"].append(cytotox_model["best_model"])
        
        # Future steps (when modules are added)
        future_steps = [
            {
                "step": 3,
                "task": "Toxicity & Safety Assessment",
                "model": "toxicity-predictor",
                "purpose": "Predict adverse effects and safety profile",
                "output": "Toxicity risk scores",
                "status": "🔮 Future module"
            },
            {
                "step": 4,
                "task": "Clinical Trial Prediction",
                "model": "clinical-predictor",
                "purpose": "Predict clinical trial outcomes",
                "output": "Success probability and side effects",
                "status": "🔮 Future module"
            },
            {
                "step": 5,
                "task": "Molecular Optimization",
                "model": "generative-optimizer",
                "purpose": "Generate improved molecules",
                "output": "Optimized compound variants",
                "status": "🔮 Future module"
            }
        ]
        
        workflow["future_steps"] = future_steps
        
        return workflow
    
    def execute_prediction_workflow(self, workflow: Dict):
        """Execute the planned workflow (demo)"""
        print(f"\n🚀 {self.name}: Executing drug discovery workflow...")
        
        for step_info in workflow["steps"]:
            step = step_info["step"]
            task = step_info["task"]
            model = step_info["model"]
            
            print(f"\n   Step {step}: {task}")
            print(f"   🤖 Using model: {model}")
            
            # Get model capabilities
            response = requests.get(f"{self.registry_url}/discover/capabilities/{model}")
            if response.status_code == 200:
                capabilities = response.json()
                print(f"   📊 Model performance: R² = {capabilities.get('performance', {}).get('r2_score', 'N/A')}")
                print(f"   🎯 Prediction types: {', '.join(capabilities.get('output_types', []))}")
            
            print(f"   ✅ {step_info['purpose']}")
            print(f"   📤 Output: {step_info['output']}")
        
        # Show future capabilities
        print(f"\n🔮 Future Workflow Steps (when modules are added):")
        for step_info in workflow.get("future_steps", []):
            step = step_info["step"]
            task = step_info["task"]
            status = step_info["status"]
            print(f"   Step {step}: {task} - {status}")

async def main():
    """Demo the AI Agent using the Model Registry"""
    print("=" * 60)
    print("🤖 AI AGENT MODEL REGISTRY DEMONSTRATION")
    print("=" * 60)
    
    # Create AI Agent
    agent = AIAgent()
    
    # Test compound (Aspirin)
    test_compound = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Plan workflow
    workflow = agent.plan_drug_discovery_workflow(test_compound)
    
    # Execute current workflow
    agent.execute_prediction_workflow(workflow)
    
    # Show workflow summary
    print(f"\n📋 WORKFLOW SUMMARY:")
    print(f"   Compound: {workflow['compound']}")
    print(f"   Current steps: {len(workflow['steps'])}")
    print(f"   Future steps: {len(workflow.get('future_steps', []))}")
    print(f"   Models used: {', '.join(workflow['models_used'])}")
    
    print(f"\n🎯 This demonstrates how AI agents can:")
    print(f"   ✅ Automatically discover available models")
    print(f"   ✅ Select best models for specific tasks")
    print(f"   ✅ Plan multi-step prediction workflows")
    print(f"   ✅ Adapt to new models as they're added")
    
    print(f"\n💡 Next: Add the 4 new prediction modules to complete the workflow!")

if __name__ == "__main__":
    asyncio.run(main())