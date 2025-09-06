"""
OpenFDA Approvals Connector
Fetches real FDA approval data with machine-auditable provenance
WITH STRICT SYNTHETIC DATA PREVENTION
"""

import requests
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
import logging
import time

logger = logging.getLogger(__name__)

OPENFDA = "https://api.fda.gov/drug/drugsfda.json"

# Synthetic data detection
SYNTHETIC_KEYWORDS = [
    'demo', 'synthetic', 'fake', 'test', 'generated', 'artificial',
    'chembl_demo', 'mock', 'placeholder', 'example'
]


def validate_drug_name_authentic(name: str) -> bool:
    """
    Validate that drug name is not synthetic
    
    Args:
        name: Drug name to validate
        
    Returns:
        True if authentic, False if synthetic
    """
    if not name or pd.isna(name):
        return False
    
    name_lower = str(name).lower()
    
    # Check for synthetic keywords
    for keyword in SYNTHETIC_KEYWORDS:
        if keyword in name_lower:
            logger.warning(f"🚫 Synthetic drug name detected: {name} (contains '{keyword}')")
            return False
    
    # Check for obvious test patterns
    if name_lower.startswith('chembl_'):
        return False  # ChEMBL IDs are not real drug names
    
    if len(name_lower) < 3:
        return False  # Too short to be real drug name
    
    return True


def _q(name: str) -> str:
    """Create openFDA search query"""
    name = name.replace('"', '\\"')
    return f'openfda.brand_name:"{name}" OR openfda.generic_name:"{name}" OR openfda.substance_name:"{name}"'


@retry(wait=wait_exponential(1, 2, 20), stop=stop_after_attempt(5))
def query_openfda_one(name: str, timeout=30):
    """
    Query openFDA for a single drug name
    
    Args:
        name: Drug name to query
        timeout: Request timeout
        
    Returns:
        List of results or None if not found
    """
    # Validate drug name is authentic before querying
    if not validate_drug_name_authentic(name):
        logger.debug(f"Skipping synthetic drug name: {name}")
        return None
    
    try:
        r = requests.get(
            OPENFDA, 
            params={"search": _q(name), "limit": 100}, 
            timeout=timeout
        )
        
        if r.status_code == 404:
            logger.debug(f"No FDA data found for: {name}")
            return None
        elif r.status_code != 200:
            logger.warning(f"FDA API error {r.status_code} for {name}")
            return None
        
        js = r.json()
        results = js.get("results", [])
        
        # Validate results are not synthetic
        if results:
            logger.debug(f"✅ Found {len(results)} FDA records for: {name}")
        
        return results
        
    except Exception as e:
        logger.warning(f"Error querying FDA for {name}: {e}")
        return None


def validate_fda_results(results, drug_name):
    """
    Validate FDA results contain no synthetic data
    
    Args:
        results: FDA API results
        drug_name: Original drug name queried
        
    Returns:
        Validated results with synthetic entries removed
    """
    if not results:
        return results
    
    clean_results = []
    
    for result in results:
        # Check for synthetic indicators in FDA data
        is_synthetic = False
        
        # Check application data
        applications = result.get('applications', [])
        for app in applications:
            app_number = str(app.get('application_number', ''))
            
            # Check for synthetic application numbers
            if any(keyword in app_number.lower() for keyword in SYNTHETIC_KEYWORDS):
                logger.warning(f"🚫 Synthetic FDA application detected: {app_number}")
                is_synthetic = True
                break
            
            # Check product data
            products = app.get('products', [])
            for product in products:
                # Check active ingredients
                ingredients = product.get('active_ingredients', [])
                for ingredient in ingredients:
                    ingredient_name = str(ingredient.get('name', ''))
                    
                    if any(keyword in ingredient_name.lower() for keyword in SYNTHETIC_KEYWORDS):
                        logger.warning(f"🚫 Synthetic ingredient detected: {ingredient_name}")
                        is_synthetic = True
                        break
                
                if is_synthetic:
                    break
            
            if is_synthetic:
                break
        
        if not is_synthetic:
            clean_results.append(result)
        else:
            logger.warning(f"🚫 Removed synthetic FDA result for: {drug_name}")
    
    return clean_results


def extract_first_approval(results):
    """
    Extract first approval date and application number from FDA results
    
    Args:
        results: Validated FDA API results
        
    Returns:
        Tuple of (first_approval_date, application_number)
    """
    if not results:
        return None, None
    
    # Iterate applications → products → approvals
    dates = []
    appnos = set()
    
    for r in results:
        for app in r.get("applications", []):
            appno = app.get("application_number")
            
            for prod in app.get("products", []):
                # approvals may live under 'approvals' or 'original_approvals'
                approvals = prod.get("approvals", []) or prod.get("original_approvals", [])
                
                for ap in approvals:
                    d = ap.get("approval_date")
                    if d:
                        # Validate approval date is not obviously fake
                        try:
                            # Basic date format validation
                            if len(d) >= 8 and '-' in d:
                                dates.append(d)
                                if appno:
                                    appnos.add(appno)
                        except Exception:
                            logger.debug(f"Invalid date format: {d}")
                            continue
    
    if not dates:
        return None, None
    
    dates_sorted = sorted(dates)
    first_date = dates_sorted[0]
    first_appno = sorted(appnos)[0] if appnos else None
    
    return first_date, first_appno


def fetch_approvals_for_names(names: list[str]) -> pd.DataFrame:
    """
    Fetch FDA approvals for list of drug names
    
    Args:
        names: List of drug names to query
        
    Returns:
        DataFrame with approval data
    """
    logger.info(f"📡 FETCHING FDA APPROVALS FOR {len(names):,} DRUGS")
    logger.info("🚫 Strict synthetic data prevention enabled")
    logger.info("=" * 50)
    
    rows = []
    successful_queries = 0
    synthetic_skipped = 0
    
    for i, nm in enumerate(names, 1):
        try:
            # Validate drug name is authentic
            if not validate_drug_name_authentic(nm):
                synthetic_skipped += 1
                continue
            
            # Query openFDA
            logger.debug(f"[{i:4d}/{len(names)}] Querying: {nm}")
            
            res = query_openfda_one(nm)
            
            if res:
                # Validate results are authentic
                clean_res = validate_fda_results(res, nm)
                
                if clean_res:
                    first_date, appno = extract_first_approval(clean_res)
                    
                    if first_date:
                        rows.append({
                            "query_name": nm,
                            "approval_date_first": first_date,
                            "approval_source": f"openfda:{appno}" if appno else "openfda:unknown",
                            "approval_region": "US"
                        })
                        successful_queries += 1
                        logger.debug(f"  ✅ Found approval: {first_date}")
            
            # Rate limiting
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(names)} ({successful_queries} approvals found)")
                time.sleep(1)  # Brief pause every 50 queries
            
        except Exception as e:
            logger.warning(f"Error processing {nm}: {e}")
            continue
    
    logger.info(f"✅ FDA approval fetching complete:")
    logger.info(f"   Total queries: {len(names):,}")
    logger.info(f"   Synthetic names skipped: {synthetic_skipped:,}")
    logger.info(f"   Successful approvals: {successful_queries:,}")
    logger.info(f"   Success rate: {successful_queries/len(names)*100:.1f}%")
    
    return pd.DataFrame(rows)