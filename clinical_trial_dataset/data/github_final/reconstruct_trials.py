#!/usr/bin/env python3
"""Reconstruct Complete Clinical Trials Dataset"""
import pandas as pd

def reconstruct_trials():
    parts = []
    for i in range(1, 5):
        df = pd.read_csv(f"trials_part_{i}.csv")
        parts.append(df)
    
    complete = pd.concat(parts, ignore_index=True)
    complete.to_csv("complete_clinical_trials_reconstructed.csv", index=False)
    
    print(f"✅ Reconstructed: {len(complete):,} trials")
    
    # Verify NCT02688101
    nct = complete[complete['nct_id'] == 'NCT02688101']
    print(f"🎯 NCT02688101: {'✅ Found' if len(nct) > 0 else '❌ Missing'}")
    
    return complete

if __name__ == "__main__":
    reconstruct_trials()
