"""
Find working Tox21 cytotoxicity AIDs from PubChem
"""

import requests
import time
import json

def test_aid(aid):
    """Test if AID exists and has CSV data"""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/summary/JSON"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            assay_info = data.get('AssaySummaries', {}).get('AssaySummary', [])
            if assay_info:
                name = assay_info[0].get('Name', '')
                description = assay_info[0].get('Description', [])
                
                # Check if it's cytotoxicity related
                full_text = (name + ' ' + ' '.join(description)).lower()
                
                if any(keyword in full_text for keyword in ['cytotoxicity', 'cytotoxic', 'cell viability', 'cell death', 'viability']):
                    print(f"✅ AID {aid}: {name}")
                    
                    # Test CSV availability
                    csv_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
                    csv_response = requests.head(csv_url, timeout=10)
                    
                    if csv_response.status_code == 200:
                        size = csv_response.headers.get('content-length', 0)
                        print(f"   💾 CSV available, size: {size} bytes")
                        return aid, name, size
                    else:
                        print(f"   ❌ CSV not available: {csv_response.status_code}")
                else:
                    print(f"⚪ AID {aid}: Not cytotoxicity related - {name}")
        else:
            print(f"❌ AID {aid}: Not found")
        
    except Exception as e:
        print(f"❌ AID {aid}: Error - {e}")
    
    time.sleep(0.2)  # Rate limiting
    return None

# Test potential Tox21 cytotoxicity AIDs
potential_aids = [
    # Original AIDs from current work
    1852, 1853,
    
    # Tox21 range 1
    720516, 720517, 720518, 720519, 720520,
    720521, 720522, 720523, 720524, 720525,
    
    # Tox21 range 2
    743040, 743041, 743042, 743043, 743044,
    743045, 743046, 743047, 743048, 743049,
    
    # Alternative range
    588340, 588341, 588342, 588343, 588344,
    588345, 588346, 588347, 588348, 588349,
    
    # Known working range
    485290, 485291, 485292, 485293, 485294,
    485295, 485296, 485297, 485298, 485299
]

print("🧬 SEARCHING FOR WORKING TOX21 CYTOTOXICITY AIDS")
print("=" * 60)

working_aids = []

for aid in potential_aids:
    result = test_aid(aid)
    if result:
        working_aids.append(result)
        if len(working_aids) >= 5:  # Get 5 working AIDs
            break

print(f"\n🎯 FOUND {len(working_aids)} WORKING CYTOTOXICITY AIDS:")
for aid, name, size in working_aids:
    print(f"  • AID {aid}: {name} (Size: {size} bytes)")

if working_aids:
    print(f"\n✅ Use these AIDs in the extractor:")
    aids_dict = {}
    for i, (aid, name, size) in enumerate(working_aids):
        clean_name = name.replace(' ', '_').replace(':', '').replace(',', '').replace('-', '_')[:50]
        aids_dict[aid] = {
            'name': clean_name,
            'description': name,
            'category': 'cytotoxicity',
            'priority': 1 if i < 2 else 2
        }
    
    print(json.dumps(aids_dict, indent=2))
else:
    print("❌ No working cytotoxicity AIDs found")