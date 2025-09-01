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

# Logging configuration
LOG_LEVEL = "INFO"

# Quality control parameters
MIN_ENROLLMENT_COUNT = 1
MAX_ENROLLMENT_COUNT = 10000
MIN_MOLECULAR_WEIGHT = 100
MAX_MOLECULAR_WEIGHT = 2000
MIN_TRIAL_DURATION_DAYS = 30
MAX_TRIAL_DURATION_DAYS = 3650  # 10 years

# Feature engineering parameters
MAX_TFIDF_FEATURES = 100
MIN_CATEGORY_FREQUENCY = 10

# Train/test split parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42