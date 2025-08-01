"""
🚀 SIMPLE MODAL DEPLOYMENT 
Just provide your Modal API credentials and deploy instantly!
"""

def deploy_molbert_to_modal(modal_token_id, modal_token_secret):
    """
    Deploy MolBERT training to Modal with your API credentials
    
    Args:
        modal_token_id: Your Modal token ID from https://modal.com/settings/tokens  
        modal_token_secret: Your Modal token secret
    
    Returns:
        dict: Deployment status and information
    """
    
    import os
    import sys
    sys.path.append('/app/modal_training')
    
    try:
        # Set Modal credentials
        os.environ['MODAL_TOKEN_ID'] = modal_token_id
        os.environ['MODAL_TOKEN_SECRET'] = modal_token_secret
        
        print("🔑 Modal credentials set")
        print("🚀 Starting deployment...")
        
        # Import Modal components
        import modal
        from modal_molbert import app, train_all_targets
        
        # Deploy the app
        print("📦 Deploying Modal app...")
        with modal.enable_output():
            app.deploy()
        
        print("✅ App deployed!")
        
        # Start training for all targets
        print("🎯 Starting training for all 6 targets...")
        webhook_url = "https://0eb557dc-32db-47bd-b857-78931c31ecdb.preview.emergentagent.com/api/gpu/training-progress"
        
        result = train_all_targets.remote(
            targets=["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"],
            max_epochs=50,
            batch_size=64,
            webhook_url=webhook_url
        )
        
        print("🎉 SUCCESS! Training is now running on Modal GPU!")
        print("📊 Expected completion: 6-12 hours")
        print("💰 Expected cost: ~$12-24")  
        print("📈 Expected R² improvement: 0.018 → 0.6+")
        print("🔗 Monitor at: https://modal.com/apps")
        print(f"📡 Progress updates: {webhook_url}")
        
        return {
            "status": "success",
            "message": "MolBERT training deployed to Modal GPU",
            "targets": ["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"],
            "monitor_url": "https://modal.com/apps",
            "progress_url": webhook_url,
            "expected_duration_hours": "6-12",
            "expected_cost_usd": "12-24"
        }
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "message": "Check your Modal API credentials and try again"
        }

def get_modal_credentials_help():
    """Show how to get Modal API credentials"""
    print("""
🔑 GET YOUR MODAL API CREDENTIALS:

1. Go to: https://modal.com/settings/tokens
2. Click "Create Token" 
3. Copy both:
   - Token ID (starts with "ak-")
   - Token Secret (longer string)

Then call:
deploy_molbert_to_modal(
    modal_token_id="ak-your-token-id",
    modal_token_secret="your-token-secret"
)
""")

if __name__ == "__main__":
    print("🚀 Modal MolBERT Deployment Ready!")
    get_modal_credentials_help()