"""
Expanded Multi-Source Data Extraction Strategy
Target: 200K-500K bioactivity records from 15+ sources
"""

import requests
import pandas as pd
import json
from typing import Dict, List, Any
import time
from pathlib import Path

# EXPANDED DATA SOURCES CONFIGURATION
EXPANDED_DATA_SOURCES = {
    # Tier 1: Large Public Databases
    'chembl': {
        'priority': 1,
        'api_url': 'https://www.ebi.ac.uk/chembl/api/data',
        'expected_records': 30000,
        'rate_limit': 0.2
    },
    'pubchem_bioassay': {
        'priority': 1,
        'api_url': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug',
        'expected_records': 50000,
        'rate_limit': 0.3
    },
    'bindingdb': {
        'priority': 1,
        'api_url': 'https://www.bindingdb.org/bind/chemsearch/marvin/',
        'expected_records': 25000,
        'rate_limit': 0.5
    },
    'dtc': {
        'priority': 1,
        'api_url': 'https://drugtargetcommons.fimm.fi/api/',
        'expected_records': 15000,
        'rate_limit': 0.4
    },
    
    # Tier 2: Specialized Sources
    'pdb_ligands': {
        'priority': 2,
        'api_url': 'https://www.rcsb.org/pdb/rest/',
        'expected_records': 8000,
        'rate_limit': 0.3
    },
    'drugbank': {
        'priority': 2,
        'api_url': 'https://go.drugbank.com/releases/',
        'expected_records': 12000,
        'rate_limit': 1.0  # More restrictive
    },
    'iuphar': {
        'priority': 2,
        'api_url': 'https://www.guidetopharmacology.org/services/',
        'expected_records': 20000,
        'rate_limit': 0.5
    },
    
    # Tier 3: Disease-Specific
    'gdsc': {
        'priority': 3,
        'api_url': 'https://www.cancerrxgene.org/api/',
        'expected_records': 15000,
        'rate_limit': 0.4
    },
    'ccle': {
        'priority': 3,
        'api_url': 'https://sites.broadinstitute.org/ccle/',
        'expected_records': 10000,
        'rate_limit': 0.6
    },
    'pdsp': {
        'priority': 3,
        'api_url': 'https://pdsp.unc.edu/databases/',
        'expected_records': 8000,
        'rate_limit': 0.8
    },
    
    # Tier 4: Literature Mining
    'patent_mining': {
        'priority': 4,
        'expected_records': 30000,
        'rate_limit': 2.0  # Slower due to text processing
    },
    'pubmed_mining': {
        'priority': 4,
        'api_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
        'expected_records': 25000,
        'rate_limit': 1.0
    }
}

# TARGET DATASET SIZE PROJECTION
def calculate_expanded_dataset_size():
    """Calculate expected dataset size from all sources"""
    
    total_expected = 0
    sources_breakdown = {}
    
    for source, config in EXPANDED_DATA_SOURCES.items():
        expected = config['expected_records']
        total_expected += expected
        sources_breakdown[source] = expected
    
    # Apply deduplication factor (estimate 30% overlap)
    dedup_factor = 0.7
    final_expected = int(total_expected * dedup_factor)
    
    print("🎯 EXPANDED DATASET PROJECTION")
    print("=" * 50)
    print(f"Raw total across all sources: {total_expected:,}")
    print(f"After deduplication (30% overlap): {final_expected:,}")
    print()
    print("📊 Source Breakdown:")
    for source, count in sorted(sources_breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_expected) * 100
        tier = EXPANDED_DATA_SOURCES[source].get('priority', 5)
        print(f"  Tier {tier} - {source:20s}: {count:6,} records ({percentage:4.1f}%)")
    
    return final_expected, sources_breakdown

def create_extraction_plan():
    """Create phased extraction plan"""
    
    total_expected, breakdown = calculate_expanded_dataset_size()
    
    # Phase extraction by priority
    phases = {
        1: {'sources': [], 'expected': 0, 'duration': '2-3 hours'},
        2: {'sources': [], 'expected': 0, 'duration': '3-4 hours'},
        3: {'sources': [], 'expected': 0, 'duration': '4-6 hours'},
        4: {'sources': [], 'expected': 0, 'duration': '6-8 hours'}
    }
    
    for source, config in EXPANDED_DATA_SOURCES.items():
        priority = config['priority']
        phases[priority]['sources'].append(source)
        phases[priority]['expected'] += config['expected_records']
    
    print("\n📋 PHASED EXTRACTION PLAN")
    print("=" * 50)
    
    cumulative = 0
    for phase, info in phases.items():
        cumulative += info['expected']
        print(f"\nPhase {phase}: {info['duration']}")
        print(f"  Sources: {len(info['sources'])}")
        print(f"  Expected records: {info['expected']:,}")
        print(f"  Cumulative: {cumulative:,}")
        print(f"  Sources: {', '.join(info['sources'])}")
    
    print(f"\n🎯 FINAL TARGET: {total_expected:,} → {int(total_expected * 0.7):,} (deduplicated)")
    
    return phases

def estimate_training_readiness():
    """Estimate if expanded dataset will be sufficient for PropMolFlow"""
    
    expected_size, _ = calculate_expanded_dataset_size()
    
    # Analysis based on molecular generation literature
    assessments = {
        'dataset_size': {
            'current': 25000,
            'expanded': expected_size,
            'minimum_recommended': 100000,
            'ideal': 500000
        },
        'diversity_analysis': {
            'targets': 23,
            'avg_per_target_current': 25000 // 23,
            'avg_per_target_expanded': expected_size // 23,
            'recommended_per_target': 5000
        }
    }
    
    print("\n📊 TRAINING READINESS ASSESSMENT")
    print("=" * 50)
    
    size_score = min(expected_size / 100000, 1.0) * 100
    diversity_score = min((expected_size // 23) / 5000, 1.0) * 100
    
    print(f"Dataset Size Score: {size_score:.0f}/100")
    print(f"  Current: {assessments['dataset_size']['current']:,}")
    print(f"  Expanded: {assessments['dataset_size']['expanded']:,}")
    print(f"  Minimum: {assessments['dataset_size']['minimum_recommended']:,}")
    print(f"  Status: {'✅ Sufficient' if size_score >= 80 else '⚠️ Marginal' if size_score >= 60 else '❌ Insufficient'}")
    
    print(f"\nDiversity Score: {diversity_score:.0f}/100")
    print(f"  Avg per target (current): {assessments['diversity_analysis']['avg_per_target_current']:,}")
    print(f"  Avg per target (expanded): {assessments['diversity_analysis']['avg_per_target_expanded']:,}")
    print(f"  Recommended per target: {assessments['diversity_analysis']['recommended_per_target']:,}")
    print(f"  Status: {'✅ Good diversity' if diversity_score >= 80 else '⚠️ Moderate' if diversity_score >= 60 else '❌ Low diversity'}")
    
    overall_score = (size_score + diversity_score) / 2
    print(f"\n🎯 OVERALL TRAINING READINESS: {overall_score:.0f}/100")
    
    if overall_score >= 80:
        print("✅ RECOMMENDATION: Proceed with PropMolFlow training")
    elif overall_score >= 60:
        print("⚠️ RECOMMENDATION: Proceed with caution, consider augmentation")
    else:
        print("❌ RECOMMENDATION: Need more data sources or augmentation")
    
    return overall_score

if __name__ == "__main__":
    calculate_expanded_dataset_size()
    create_extraction_plan()
    estimate_training_readiness()