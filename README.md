# 🧩 customer-agent

A public pipeline for building shopping-agent training data and evaluating retrieval/tool-use models. It covers pseudo-data generation, SFT/RLVR model training with [verl](https://github.com/volcengine/verl), and evaluation.

## Public Data Build Pipeline

> **Overview:**
> Follow the sequential pipeline to build public data from pseudo data generation through model training and evaluation.

1. **Pseudo_Data_Generation_1**
   - Read ESCI dataset and generate shopping trajectories using Claude
   - Filter noisy data and verify action distributions
   - Construct verifiable question-answer using GPT-oss

2. **Pseudo_Data_Generation_2**
   - Generate multi-turn SFT data
   - Build SFT and RLVR training datasets
   - Move processed data to training directory

3. **Public_Model_Training**
   - Set up Docker containers for Verl training and sandbox
   - Run SFT and RLVR training on multi-GPU setup
   - Merge model checkpoints if needed

4. **Public_Retrieve_Evaluation**
   - Run inference using Bedrock API or local GPU
   - Generate evaluation results

5. **Retool_Evaluation**
   - Tool-use evaluation on math benchmarks (AIME24/AIME25)

## Datasets

The training and evaluation datasets are published on Hugging Face:
[`hongyeeliu/ShopTrajQA`](https://huggingface.co/datasets/hongyeeliu/ShopTrajQA)

## Credentials

This project calls model APIs (e.g. Amazon Bedrock). **No credentials are committed to
this repo.** Provide them via environment variables at runtime, for example:

```bash
export AWS_BEARER_TOKEN_BEDROCK="<your-bedrock-api-key>"
```

## Evaluation

```bash
cd Retool_Evaluation
pip install "git+https://github.com/tongyx361/symeval.git"
pip install timeout_decorator
bash scripts/eval.sh
```
