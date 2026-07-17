## 🔄 SFT and RLVR Data Construction

> **Overview:**  
> Build SFT and RLVR training data, then move to the model training directory `Public_Model_Training`.

### Build Training Data

```bash
# Step 1: Build SFT data
python convert_neo_multi_sft.py

# Step 2: Build RLVR data
python convert_neo_multi_rlvr.py
```

### Move Data to Training Directory

```bash
# Step 3: Move processed data
cp data ../../Public_Model_Training
```

### Optional: Merge Multiple Files with Unique QID

> **Note:**  
> Only required if you have multiple files that need to be merged while processing with unique question IDs.

```bash
python combine_with_unique_qid.py
```

Refer to `combine_with_unique_qid.py` for implementation details.