"""
S3 Manager for Model Registry
Handles S3 operations for model artifacts and datasets
"""
import os
import boto3
import hashlib
import logging
from typing import Optional, Tuple
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class S3Manager:
    """Manages S3 operations for model registry"""
    
    def __init__(self):
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = os.getenv("S3_BUCKET", "veridicabatabase")
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials not found in environment variables")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )
    
    def test_connection(self) -> bool:
        """Test S3 connection and bucket accessibility"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ S3 connection successful. Bucket: {self.bucket_name}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"❌ S3 connection failed: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_key: str) -> Tuple[str, str, int]:
        """
        Upload file to S3 and return URI, SHA256, size
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (path within bucket)
            
        Returns:
            Tuple of (s3_uri, sha256_hash, file_size_bytes)
        """
        try:
            # Calculate SHA256 hash
            sha256_hash = self._calculate_sha256(local_path)
            
            # Get file size
            file_size = os.path.getsize(local_path)
            
            # Upload to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            # Construct S3 URI
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            
            logger.info(f"✅ Uploaded {local_path} to {s3_uri}")
            return s3_uri, sha256_hash, file_size
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {local_path}: {e}")
            raise
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            Success boolean
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"✅ Downloaded {s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download {s3_key}: {e}")
            return False
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for S3 object
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"❌ Failed to generate presigned URL for {s3_key}: {e}")
            return None
    
    def object_exists(self, s3_key: str) -> bool:
        """Check if object exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def list_objects(self, prefix: str) -> list:
        """List objects with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return response.get('Contents', [])
        except Exception as e:
            logger.error(f"❌ Failed to list objects with prefix {prefix}: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"✅ Deleted {s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete {s3_key}: {e}")
            return False
    
    def get_s3_key_from_uri(self, s3_uri: str) -> str:
        """Extract S3 key from S3 URI"""
        if s3_uri.startswith(f"s3://{self.bucket_name}/"):
            return s3_uri[len(f"s3://{self.bucket_name}/"):]
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

# Global S3 manager instance
s3_manager = None

def get_s3_manager() -> S3Manager:
    """Get global S3 manager instance"""
    global s3_manager
    if s3_manager is None:
        s3_manager = S3Manager()
    return s3_manager