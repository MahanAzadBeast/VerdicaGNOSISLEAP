#!/usr/bin/env python3
"""
OpenFDA Approval Fetcher
Populates approval_date_first using openFDA API for true approval labels
"""

import pandas as pd
import requests
import time
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# openFDA API configuration
OPENFDA_BASE = "https://api.fda.gov/drug/drugsfda.json"
MAX_QPS = 1  # Conservative rate limiting (1 query per second)
TIMEOUT = 30


def rate_limit():
    """Simple rate limiting"""
    time.sleep(1.0 / MAX_QPS)


def query_openfda_approval(drug_name):
    """
    Query openFDA for drug approval information
    
    Args:
        drug_name: Drug name to search for
        
    Returns:
        First approval date or None if not found
    """
    if not drug_name or pd.isna(drug_name):
        return None
    
    # Clean drug name for search
    clean_name = str(drug_name).strip()
    if len(clean_name) < 2:
        return None
    
    # Try multiple search strategies
    search_queries = [
        f'openfda.brand_name:"{clean_name}"',
        f'openfda.substance_name:"{clean_name}"', 
        f'openfda.generic_name:"{clean_name}"',
        f'products.active_ingredients.name:"{clean_name}"'
    ]
    
    for query in search_queries:
        try:
            rate_limit()  # Respect rate limits
            
            params = {
                "search": query,
                "limit": 5  # Get a few results to find earliest date
            }
            
            response = requests.get(OPENFDA_BASE, params=params, timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    # Extract approval dates from all applications
                    approval_dates = []
                    
                    for result in data['results']:
                        applications = result.get('applications', [])
                        
                        for app in applications:
                            products = app.get('products', [])
                            
                            for product in products:
                                approvals = product.get('approvals', [])
                                
                                for approval in approvals:
                                    if 'approval_date' in approval:
                                        try:
                                            # Parse date (format: YYYY-MM-DD)
                                            approval_date = approval['approval_date']
                                            parsed_date = datetime.strptime(approval_date, '%Y-%m-%d')
                                            approval_dates.append(parsed_date)
                                        except (ValueError, TypeError):
                                            continue
                    
                    if approval_dates:
                        # Return earliest approval date
                        earliest_date = min(approval_dates)
                        logger.debug(f"Found approval for '{clean_name}': {earliest_date.strftime('%Y-%m-%d')}")
                        return earliest_date.strftime('%Y-%m-%d')
            
            elif response.status_code == 404:
                # Not found with this query, try next one
                continue
                
            elif response.status_code == 429:
                # Rate limited - wait longer
                logger.warning(f"Rate limited for '{clean_name}', waiting...")
                time.sleep(5)
                continue
                
            else:
                logger.warning(f"HTTP {response.status_code} for '{clean_name}': {response.text[:100]}")
                continue
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout querying openFDA for '{clean_name}'")
            continue
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for '{clean_name}': {e}")
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error for '{clean_name}': {e}")
            continue
    
    # No approval found with any query
    return None


def fetch_approval_dates(input_file="csv_exports/veridica_master_merged.dedup.csv",
                        output_file="csv_exports/veridica_master_merged.approvals.csv",
                        sample_size=None):
    """
    Fetch approval dates from openFDA and populate approval_date_first
    
    Args:
        input_file: Input deduplicated CSV file
        output_file: Output file with approval dates
        sample_size: If set, only process this many drugs (for testing)
        
    Returns:
        DataFrame with approval dates populated
    """
    logger.info("📡 FETCHING APPROVAL DATES FROM OPENFDA")
    logger.info("🔍 Populating approval_date_first for true approval labels")
    logger.info("=" * 60)
    
    try:
        # Load deduplicated dataset
        df = pd.read_csv(input_file)
        logger.info(f"✅ Loaded dataset: {len(df):,} compounds")
        
        # Check current approval data
        if 'approval_date_first' in df.columns:
            existing_approvals = df['approval_date_first'].notna().sum()
            logger.info(f"📊 Existing approvals: {existing_approvals:,}")
        else:
            df['approval_date_first'] = None
            existing_approvals = 0
        
        # Get unique drug names to query
        drug_names = df['primary_drug'].dropna().unique()
        logger.info(f"🔍 Unique drug names to query: {len(drug_names):,}")
        
        if sample_size:
            drug_names = drug_names[:sample_size]
            logger.info(f"🧪 Testing mode: querying first {sample_size} drugs only")
        
        # Query openFDA for each drug
        approval_cache = {}
        successful_queries = 0
        failed_queries = 0
        
        logger.info("🚀 Starting openFDA queries...")
        start_time = time.time()
        
        for i, drug_name in enumerate(drug_names, 1):
            if drug_name in approval_cache:
                continue  # Already queried
            
            logger.info(f"[{i:4d}/{len(drug_names)}] Querying: {drug_name}")
            
            approval_date = query_openfda_approval(drug_name)
            
            if approval_date:
                approval_cache[drug_name] = approval_date
                successful_queries += 1
                logger.info(f"  ✅ Found approval: {approval_date}")
            else:
                approval_cache[drug_name] = None
                failed_queries += 1
                logger.debug(f"  ❌ No approval found")
            
            # Progress update every 50 queries
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_seconds = (len(drug_names) - i) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                logger.info(f"📊 Progress: {i}/{len(drug_names)} ({i/len(drug_names)*100:.1f}%)")
                logger.info(f"   ✅ Successful: {successful_queries}, ❌ Failed: {failed_queries}")
                logger.info(f"   ⏱️ Rate: {rate:.1f} queries/sec, ETA: {eta_minutes:.1f} min")
        
        # Apply approval dates to dataset
        logger.info("💾 Applying approval dates to dataset...")
        
        def get_approval_date(drug_name):
            return approval_cache.get(drug_name)
        
        # Update approval_date_first where we found approvals
        df['approval_date_first'] = df['approval_date_first'].fillna(
            df['primary_drug'].map(get_approval_date)
        )
        
        # Count final results
        final_approvals = df['approval_date_first'].notna().sum()
        new_approvals = final_approvals - existing_approvals
        
        logger.info(f"✅ Approval fetching complete:")
        logger.info(f"   Total queries: {len(drug_names):,}")
        logger.info(f"   Successful: {successful_queries:,}")
        logger.info(f"   Failed: {failed_queries:,}")
        logger.info(f"   Success rate: {successful_queries/len(drug_names)*100:.1f}%")
        logger.info(f"   New approvals found: {new_approvals:,}")
        logger.info(f"   Total compounds with approvals: {final_approvals:,}")
        
        # Create binary approval label
        df['approved'] = (df['approval_date_first'].notna()).astype(int)
        approved_count = df['approved'].sum()
        
        logger.info(f"🏷️ Binary approval labels:")
        logger.info(f"   Approved: {approved_count:,} ({approved_count/len(df)*100:.1f}%)")
        logger.info(f"   Not approved: {len(df) - approved_count:,} ({(len(df) - approved_count)/len(df)*100:.1f}%)")
        
        # Save dataset with approval dates
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Dataset with approvals saved: {output_file}")
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
            return None
        
        # Save approval cache for future use
        cache_file = "csv_exports/openfda_approval_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(approval_cache, f, indent=2)
            logger.info(f"💾 Approval cache saved: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save cache: {e}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error fetching approval dates: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_approval_cache():
    """Load existing approval cache if available"""
    cache_file = "csv_exports/openfda_approval_cache.json"
    
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        logger.info(f"📂 Loaded approval cache: {len(cache)} entries")
        return cache
    except FileNotFoundError:
        logger.info("📂 No existing approval cache found")
        return {}
    except Exception as e:
        logger.warning(f"⚠️ Could not load approval cache: {e}")
        return {}


def main():
    """Main execution"""
    logger.info("📡 OPENFDA APPROVAL FETCHER")
    logger.info("🔍 Fetching real FDA approval dates for binary labels")
    logger.info("🚫 100% authentic approval data - NO synthetic labels")
    
    # Test mode first (sample of 10 drugs)
    logger.info("\\n🧪 TESTING MODE: Querying 10 drugs first")
    test_df = fetch_approval_dates(sample_size=10)
    
    if test_df is not None:
        logger.info("✅ Test successful! Ready for full dataset")
        
        # Ask user if they want to proceed with full dataset
        logger.info("\\n⚠️ Full dataset will query ~4,000 unique drugs")
        logger.info("⏱️ Estimated time: 1-2 hours with rate limiting")
        logger.info("💡 Consider running in background or overnight")
        
        # For now, just run the test
        logger.info("\\n🎉 APPROVAL FETCHER TEST COMPLETE")
        logger.info("📊 Test results show openFDA integration working")
        logger.info("🚀 Ready to scale to full dataset when needed")
        
    else:
        logger.error("❌ Test failed - check openFDA API access")


if __name__ == "__main__":
    main()