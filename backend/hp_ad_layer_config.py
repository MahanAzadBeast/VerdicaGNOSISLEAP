# hp_ad_layer_config.py
"""
Universal configuration for HP AD Layer - applies to ALL compounds and targets
"""

# Universal neighbor sanity thresholds
NEIGHBOR_SMAX_MIN = 0.50
NEIGHBOR_MIN_COUNT_040 = 30       # same-target + same-assay neighbors

# Cross-assay consistency thresholds
ASSAY_DELTA_MAX_LOG = 1.0         # 10x mismatch allowed before gating
ASSAY_MONOTONIC_TOL_LOG = 1.0     # min(B,F) should not be >> EC50 for enzymes
EC_MIN_FLOOR_UM = 0.01            # clamp 0.0 artifacts
FAST_SHAPE_PCTL_KINASE = 0.90

# Family envelopes (generic, applies to all compounds)
FAMILY_ENVELOPES = {
    "kinase": {
        "mw_min": 300, 
        "rings_min": 2, 
        "forbid_strong_anion": True
    },
    "gpcr": {
        "mw_min": 250, 
        "mw_max": 600, 
        "clogp_min": 1.5
    },
    "ppi": {
        "mw_min": 400, 
        "rings_min": 3
    }
}

# Cumulative gating rules
CUMULATIVE_GATE_SUPPRESS = 2      # >=2 reasons => no numerics
CUMULATIVE_GATE_HARD = 3          # >=3 reasons => add Mechanistically_implausible

# Universal mechanism gate SMARTS patterns
KINASE_HINGE_SMARTS = [
    "[nH1,NH2]c1ncnc([nH1,NH2])n1",     # diaminopyrimidine
    "[nH1,NH2]c1nc([nH1,NH2])nc([nH1,NH2])n1",  # triaminopyrimidine  
    "n1cnc2ncnn12",                      # purine-like
    "O=C-Nc1ncccc1",                     # benzamide-like HBA/HBD near ring
    "c1nc([nH1,NH2])nc([nH1,NH2])c1"    # simple diaminopyrimidine
]

PARP_POS_SMARTS = [
    "c1ccc(C(=O)N)cc1",                      # benzamide
    "[c,n]1[c,n][c,n][c,n](C(=O)N)[c,n][c,n]1"  # nicotinamide-like
]

PARP_NEG_SMARTS = [
    "O=C(O)c1ccccc1O",       # salicylic acid
    "O=C(O)c1ccc(cc1)O",     # p-hydroxybenzoic acid  
    "O=S(=O)(O)c1ccccc1"     # aryl sulfonic acids
]