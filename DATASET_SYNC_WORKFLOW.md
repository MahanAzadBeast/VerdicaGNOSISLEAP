# 🔄 GNOSIS Dataset Sync Workflow
## Modal → S3 Transfer with Audit-Ready Lineage

### 🎯 **Objective**
Sync curated datasets, deterministic splits, and eval reports from Modal to S3 with checksums and metadata for reproducibility. Establish S3 as source of truth for both training and inference.

---

## 📋 **Phase 1: Setup (One-time)**

### 1. Setup Modal AWS Secrets
```bash
cd /app/modal_training
python setup_aws_secrets.py
```
**Output**: Creates `aws-credentials` secret in Modal

### 2. Verify Modal Volumes
```bash
modal volume list
```
**Expected**: `veridica-datasets` and `veridica-models` volumes

---

## 🚀 **Phase 2: Execute Sync**

### 1. Run Modal → S3 Sync Job
```bash
cd /app/modal_training
modal run sync_datasets_to_s3.py
```

**What it syncs:**
- ✅ **Curated datasets** (processed ChEMBL, BindingDB, GDSC)
- ✅ **Deterministic splits** (train/val/test)
- ✅ **Evaluation reports** (model performance metrics)
- ✅ **SHA256 checksums** (integrity verification)
- ✅ **Metadata files** (provenance tracking)

**S3 Structure Created:**
```
s3://veridicabatabase/
├── datasets/
│   ├── chembl/v32/
│   │   ├── *.csv (curated data)
│   │   └── metadata.json (checksums + provenance)
│   ├── bindingdb/v2024/
│   │   ├── *.csv (curated data)
│   │   └── metadata.json
│   ├── gdsc/v1.0/
│   │   ├── *.csv (curated data)
│   │   └── metadata.json
│   └── _sync_reports/
│       └── 20250821_120000_sync_report.json
└── models/
    ├── gnosis-i/1.0.0/reports/
    └── gnosis-ii/1.0.0/reports/
```

### 2. Update Model Registry
```bash
cd /app/backend
python model_registry/update_datasets.py
```

**What it does:**
- ✅ Reads S3 sync reports
- ✅ Updates MongoDB with real dataset metadata
- ✅ Updates YAML manifests with file counts & checksums
- ✅ Verifies S3 integrity

---

## 📊 **Phase 3: Verification**

### 1. Check Registry Status
```bash
curl -s "http://localhost:8001/api/registry/stats" | jq '.'
```

### 2. Verify Dataset Lineage
```bash
curl -s "http://localhost:8001/api/registry/discover/summary" | jq '.datasets'
```

### 3. Test S3 Access
```bash
# List synced datasets
aws s3 ls s3://veridicabatabase/datasets/ --recursive

# Download a metadata file
aws s3 cp s3://veridicabatabase/datasets/chembl/v32/metadata.json /tmp/
```

---

## ✅ **Expected Results**

**Before Sync:**
```json
{
  "models_count": 2,
  "datasets_count": 0,  // ← No real datasets
  "artifacts_count": 2
}
```

**After Sync:**
```json
{
  "models_count": 2,
  "datasets_count": 3,  // ← Real datasets registered
  "artifacts_count": 2,
  "total_dataset_size_mb": 750,
  "s3_objects": 15
}
```

**Audit Trail Created:**
1. **SHA256 checksums** for every file
2. **Sync timestamps** and provenance
3. **File-level metadata** (size, type, path)
4. **Model-dataset lineage** mapping
5. **Reproducible splits** with deterministic seeds

---

## 🔄 **Ongoing Workflow**

### For New Datasets:
1. Process on Modal volumes
2. Run sync job: `modal run sync_datasets_to_s3.py`
3. Update registry: `python update_datasets.py`

### For Training/Inference:
- **Source of truth**: S3 datasets (not Modal volumes)
- **Verification**: Check SHA256 before training
- **Lineage**: Track which S3 dataset version was used

---

## 🎯 **Benefits Achieved**

✅ **Audit-ready lineage**: Every model → dataset → file mapping  
✅ **Reproducible splits**: Deterministic train/val/test  
✅ **Integrity verification**: SHA256 checksums  
✅ **Source of truth**: S3 for both training & inference  
✅ **Cost efficient**: Only curated data, no raw dumps  
✅ **Scalable**: Supports 100+ datasets with metadata  

---

## 🚨 **Troubleshooting**

**Modal volume not found:**
```bash
modal volume create veridica-datasets
modal volume create veridica-models
```

**AWS credentials error:**
```bash
# Re-run secret setup
python setup_aws_secrets.py
modal secret list  # Verify 'aws-credentials' exists
```

**No datasets found on Modal:**
```bash
# Check Modal volumes
modal run -c "ls -la /datasets/"
modal run -c "find /datasets/ -name '*.csv' | head -10"
```

**S3 sync incomplete:**
```bash
# Check sync report
aws s3 cp s3://veridicabatabase/datasets/_sync_reports/latest.json /tmp/
cat /tmp/latest.json | jq '.total_files'
```

---

## 📈 **Next Steps**

After successful sync:
1. ✅ **Models use S3 datasets** (not local/Modal)
2. ✅ **Training reproducibility** via checksums
3. ✅ **Ready for Phase 3**: New prediction modules
4. ✅ **Audit compliance** for regulatory submissions

**Ready to proceed with Gnosis III-V modules! 🚀**