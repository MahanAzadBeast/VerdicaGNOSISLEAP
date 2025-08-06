"""
Check all available Modal volumes for comprehensive GDSC dataset
"""

import modal
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("check-all-volumes")

image = modal.Image.debian_slim().pip_install(["pandas==2.1.0"])

# Try to access multiple known volume names
volume_names = ["expanded-datasets", "trained-models", "datasets", "gdsc-data", "cancer-data"]

@app.function(image=image, timeout=600)
def check_available_volumes():
    """Check what volumes are available and their contents"""
    
    logger.info("🔍 CHECKING ALL AVAILABLE MODAL VOLUMES")
    
    # Method 1: Try to list all volumes (this might not work but worth trying)
    try:
        from modal import Volume
        logger.info("Attempting to list all volumes...")
        # This might not work but worth a try
    except Exception as e:
        logger.info(f"Cannot list all volumes: {e}")
    
    # Method 2: Check each known volume name
    volume_results = {}
    
    for volume_name in volume_names:
        logger.info(f"🔍 Checking volume: '{volume_name}'")
        
        try:
            # Create a function to test this specific volume
            @app.function(
                image=image, 
                timeout=300,
                volumes={"/vol": modal.Volume.from_name(volume_name)}
            )
            def check_volume_content():
                try:
                    contents = os.listdir("/vol")
                    volume_info = {"exists": True, "contents": contents, "error": None}
                    
                    # Look for GDSC-related directories
                    gdsc_dirs = [item for item in contents if 'gdsc' in item.lower()]
                    if gdsc_dirs:
                        volume_info["gdsc_directories"] = gdsc_dirs
                        
                        # Check contents of GDSC directories
                        for gdsc_dir in gdsc_dirs:
                            gdsc_path = os.path.join("/vol", gdsc_dir)
                            if os.path.isdir(gdsc_path):
                                try:
                                    gdsc_contents = os.listdir(gdsc_path)
                                    csv_files = [f for f in gdsc_contents if f.endswith('.csv')]
                                    volume_info[f"{gdsc_dir}_contents"] = gdsc_contents
                                    volume_info[f"{gdsc_dir}_csv_files"] = csv_files
                                    
                                    # Check file sizes
                                    for csv_file in csv_files:
                                        csv_path = os.path.join(gdsc_path, csv_file)
                                        try:
                                            size = os.path.getsize(csv_path)
                                            volume_info[f"{csv_file}_size"] = size
                                        except:
                                            pass
                                except:
                                    volume_info[f"{gdsc_dir}_error"] = "Cannot read directory"
                    
                    return volume_info
                    
                except Exception as e:
                    return {"exists": False, "error": str(e)}
            
            # Execute the check
            result = check_volume_content.remote()
            volume_results[volume_name] = result
            
            if result["exists"]:
                logger.info(f"  ✅ Volume '{volume_name}' exists")
                logger.info(f"     Contents: {result['contents']}")
                if 'gdsc_directories' in result:
                    logger.info(f"     GDSC dirs: {result['gdsc_directories']}")
            else:
                logger.info(f"  ❌ Volume '{volume_name}' not accessible: {result['error']}")
                
        except Exception as e:
            volume_results[volume_name] = {"exists": False, "error": str(e)}
            logger.info(f"  ❌ Error checking '{volume_name}': {e}")
    
    return volume_results

if __name__ == "__main__":
    with app.run():
        results = check_available_volumes.remote()
        
        print("\n" + "="*80)
        print("📁 MODAL VOLUME ANALYSIS")
        print("="*80)
        
        accessible_volumes = 0
        gdsc_data_found = False
        
        for volume_name, result in results.items():
            print(f"\n📁 Volume: {volume_name}")
            
            if result["exists"]:
                accessible_volumes += 1
                print("  Status: ✅ ACCESSIBLE")
                print(f"  Contents: {result['contents']}")
                
                if 'gdsc_directories' in result:
                    print(f"  🎯 GDSC directories: {result['gdsc_directories']}")
                    gdsc_data_found = True
                    
                    for gdsc_dir in result['gdsc_directories']:
                        if f"{gdsc_dir}_csv_files" in result:
                            csv_files = result[f"{gdsc_dir}_csv_files"]
                            print(f"    📊 CSV files in {gdsc_dir}: {csv_files}")
                            
                            for csv_file in csv_files:
                                if f"{csv_file}_size" in result:
                                    size = result[f"{csv_file}_size"]
                                    print(f"      📄 {csv_file}: {size:,} bytes")
            else:
                print(f"  Status: ❌ NOT ACCESSIBLE ({result['error']})")
        
        print(f"\n📊 SUMMARY:")
        print(f"Accessible volumes: {accessible_volumes}")
        print(f"GDSC data found: {'✅ YES' if gdsc_data_found else '❌ NO'}")
        
        if not gdsc_data_found:
            print("\n⚠️ RECOMMENDATION:")
            print("The comprehensive GDSC dataset with >600 compounds may not be")
            print("available in the current Modal volumes. Consider:")
            print("1. Verifying the correct volume name/location")
            print("2. Re-uploading the comprehensive GDSC dataset")
            print("3. Using alternative data sources")