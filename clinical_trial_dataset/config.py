# API Configuration
CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Data collection parameters
BATCH_SIZE = 1000
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 1  # seconds
TARGET_RECORDS = 50000

# Date range for trials
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"