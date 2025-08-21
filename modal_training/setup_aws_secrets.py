#!/usr/bin/env python3
"""
Setup AWS Secrets for Modal
Create Modal secrets for S3 access
"""
import modal
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / 'backend' / '.env')

def setup_modal_aws_secrets():
    """Setup AWS credentials as Modal secrets"""
    
    print("🔐 Setting up AWS secrets for Modal...")
    
    # Get credentials from environment
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    s3_bucket = os.getenv('S3_BUCKET', 'veridicabatabase')
    
    if not aws_access_key_id or not aws_secret_access_key:
        print("❌ AWS credentials not found in environment")
        print("   Make sure your .env file contains:")
        print("   AWS_ACCESS_KEY_ID=...")
        print("   AWS_SECRET_ACCESS_KEY=...")
        return False
    
    try:
        # Create Modal secret (updated API)
        secret_dict = {
            "AWS_ACCESS_KEY_ID": aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
            "AWS_REGION": aws_region,
            "S3_BUCKET": s3_bucket
        }
        
        # Create and deploy secret
        secret = modal.Secret.from_dict(secret_dict)
        
        print("✅ AWS credentials configured for Modal")
        print(f"   Region: {aws_region}")
        print(f"   Bucket: {s3_bucket}")
        print(f"   Access Key: {aws_access_key_id[:8]}...")
        print("\n💡 Secret will be created when Modal job runs")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save Modal secret: {e}")
        return False

if __name__ == "__main__":
    setup_modal_aws_secrets()