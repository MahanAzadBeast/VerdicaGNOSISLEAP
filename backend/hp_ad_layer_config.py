# hp_ad_layer_config.py
"""
Configuration constants for HP AD Layer Cross-Assay Gating & Reliability Hardening
"""

# Neighbor sanity thresholds
NEIGHBOR_SMAX_MIN = 0.50
NEIGHBOR_MIN_COUNT_040 = 30  # same target + same assay

# Cross-assay consistency thresholds
ASSAY_DELTA_MAX_LOG = 1.0    # 10x mismatch threshold
ASSAY_MONOTONIC_TOL_LOG = 1.0

# Family physicochemical envelopes
MW_MIN_KINASE = 300
RINGS_MIN_KINASE = 2
EC50_FLOOR_UM = 0.01
FAST_SHAPE_PCTL_KINASE = 0.90

# GPCR envelopes
CLoGP_MIN_GPCR = 1.5
MW_RANGE_GPCR = (250, 600)

# PPI (Protein-Protein Interaction) envelopes
MW_MIN_PPI = 400
RINGS_MIN_PPI = 3

# Cumulative gating rules
CUMULATIVE_GATE_SUPPRESS = 2  # >=2 gates => suppress numerics
CUMULATIVE_GATE_HARD = 3      # >=3 gates => add Mechanistically_implausible

# Pharmacophore patterns
PARP_POS_SMARTS = [
    "c1ccc(C(=O)N)cc1",                      # benzamide
    "[c,n]1[c,n][c,n][c,n](C(=O)N)[c,n][c,n]1"  # nicotinamide-like
]

PARP_NEG_SMARTS = [
    "O=C(O)c1ccccc1O",                       # salicylic acid
    "O=C(O)c1ccc(cc1)O",                     # p-hydroxybenzoic acid  
    "O=S(=O)(O)c1ccccc1"                     # aryl sulfonic acids
]

KINASE_HINGE_SMARTS = [
    "n1c(N)nc(N)n1",                         # diaminopyrimidine-like
    "n1cnc2ncnc2c1",                         # purine-like
    "Nc1ncnc(N)c1",                          # hinge donors/acceptors
    "O=C-Nc1ncccc1",                         # benzamide-like HBA/HBD near ring
    "[nH1,NH1,NH2]c1nc2ccccc2c([OH,NH1,NH2])c1",  # quinazoline scaffold
    "c1cc([OH,NH1,NH2])ccc1[nH1,NH1,NH2]",       # para-substituted aniline
    "[F,Cl,Br]c1ccc(cc1)[NH1,NH1,NH2]"           # halogenated aniline
]