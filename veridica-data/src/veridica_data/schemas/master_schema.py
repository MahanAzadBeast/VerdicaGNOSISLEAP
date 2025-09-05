"""
Master Compound Table Schema
Defines the structure for the master compound index
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class MasterCompoundSchema(BaseModel):
    """Schema for master compound table - one row per compound"""
    
    # Primary keys
    chembl_id: str = Field(..., description="ChEMBL compound identifier")
    canonical_smiles: str = Field(..., description="Canonical SMILES structure")
    inchi: str = Field(..., description="InChI string")
    inchikey: str = Field(..., description="InChIKey (14-character block)")
    
    # Drug identification
    primary_drug: str = Field(..., description="Primary drug name")
    all_drug_names: Optional[list] = Field(default=None, description="All known drug names")
    
    # Molecular descriptors (complete block - no nulls allowed)
    mol_molecular_weight: float = Field(..., description="Molecular weight (Da)")
    mol_logp: float = Field(..., description="LogP (lipophilicity)")
    mol_tpsa: float = Field(..., description="Topological polar surface area")
    mol_num_hbd: int = Field(..., description="Number of hydrogen bond donors")
    mol_num_hba: int = Field(..., description="Number of hydrogen bond acceptors")
    mol_num_rotatable_bonds: int = Field(..., description="Number of rotatable bonds")
    mol_num_heavy_atoms: int = Field(..., description="Number of heavy atoms")
    mol_num_rings: int = Field(..., description="Number of rings")
    mol_num_aromatic_rings: int = Field(..., description="Number of aromatic rings")
    mol_formal_charge: int = Field(..., description="Formal charge")
    mol_num_heteroatoms: int = Field(..., description="Number of heteroatoms")
    mol_fraction_csp3: float = Field(..., description="Fraction of sp3 carbons")
    
    # Temporal fields
    first_seen_date: datetime = Field(..., description="First appearance in any database")
    source_first_seen: str = Field(..., description="Source where first seen")
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Provenance
    data_source: str = Field(..., description="Primary data source")
    compound_type: str = Field(default="Small molecule", description="Type of compound")
    
    # Quality flags
    smiles_validation_status: str = Field(default="validated", description="SMILES validation status")
    structure_standardized: bool = Field(default=True, description="Whether structure was standardized")
    has_complete_descriptors: bool = Field(default=True, description="Whether all descriptors are present")
    
    @validator('inchikey')
    def validate_inchikey(cls, v):
        """Validate InChIKey format"""
        if not v or len(v) < 14:
            raise ValueError("InChIKey must be at least 14 characters")
        return v
    
    @validator('canonical_smiles')
    def validate_smiles(cls, v):
        """Basic SMILES validation"""
        if not v or len(v) < 3:
            raise ValueError("SMILES must be at least 3 characters")
        return v
    
    @validator('mol_molecular_weight')
    def validate_molecular_weight(cls, v):
        """Validate molecular weight range"""
        if v <= 0 or v > 5000:
            raise ValueError("Molecular weight must be between 0 and 5000 Da")
        return v
    
    class Config:
        # Allow extra fields for flexibility
        extra = "allow"
        # Use enum values
        use_enum_values = True
        # Validate on assignment
        validate_assignment = True


class CompoundMapping(BaseModel):
    """Schema for compound mapping between different databases"""
    
    chembl_id: str
    pubchem_cid: Optional[int] = None
    drugbank_id: Optional[str] = None
    cas_number: Optional[str] = None
    
    # Structure keys for validation
    canonical_smiles: str
    inchikey: str
    
    # Mapping metadata
    mapping_source: str = Field(..., description="Source of the mapping")
    mapping_confidence: float = Field(..., description="Confidence score (0-1)")
    mapping_method: str = Field(..., description="Method used for mapping")
    mapped_at: datetime = Field(default_factory=datetime.now)
    
    @validator('mapping_confidence')
    def validate_confidence(cls, v):
        """Validate confidence score range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


def create_master_record(chembl_data: Dict[str, Any], 
                        canonical_smiles: str,
                        inchi: str, 
                        inchikey: str,
                        descriptors: Dict[str, float]) -> Dict[str, Any]:
    """
    Create master compound record from ChEMBL data and computed properties
    
    Args:
        chembl_data: Raw data from ChEMBL
        canonical_smiles: Standardized SMILES
        inchi: InChI string
        inchikey: InChIKey
        descriptors: Computed molecular descriptors
        
    Returns:
        Master compound record dictionary
    """
    
    return {
        # Primary keys
        "chembl_id": chembl_data.get("chembl_id"),
        "canonical_smiles": canonical_smiles,
        "inchi": inchi,
        "inchikey": inchikey,
        
        # Drug identification
        "primary_drug": chembl_data.get("primary_drug"),
        "all_drug_names": chembl_data.get("all_drug_names", []),
        
        # Molecular descriptors
        **descriptors,
        
        # Temporal fields
        "first_seen_date": datetime.now(),  # Will be updated with actual first seen
        "source_first_seen": "chembl",
        "created_at": datetime.now(),
        "last_updated": datetime.now(),
        
        # Provenance
        "data_source": "chembl_api",
        "compound_type": "Small molecule",
        
        # Quality flags
        "smiles_validation_status": "validated",
        "structure_standardized": True,
        "has_complete_descriptors": all(v is not None for v in descriptors.values())
    }


def validate_structure_consistency(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Validate that same chembl_id always maps to same inchikey
    
    Args:
        records: List of compound records
        
    Returns:
        List of inconsistency reports
    """
    chembl_to_inchikey = {}
    inconsistencies = []
    
    for record in records:
        chembl_id = record.get("chembl_id")
        inchikey = record.get("inchikey")
        
        if not chembl_id or not inchikey:
            continue
        
        if chembl_id in chembl_to_inchikey:
            if chembl_to_inchikey[chembl_id] != inchikey:
                inconsistencies.append({
                    "chembl_id": chembl_id,
                    "expected_inchikey": chembl_to_inchikey[chembl_id],
                    "found_inchikey": inchikey,
                    "error": "InChIKey mismatch for same ChEMBL ID"
                })
        else:
            chembl_to_inchikey[chembl_id] = inchikey
    
    return inconsistencies