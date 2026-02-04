# Project Execution Instructions

Here is the step-by-step guide to run the project from scratch.

## 1. Setup Environment
First, create a virtual environment and install dependencies.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install datasets  # Required for downloading data
```

## 2. Download Data
Download the raw news datasets (AG News + BBC News). This avoids sending large files manually.

```bash
python download_datasets.py --dataset all
```
*Output: Saves raw JSON files to `backend/data/raw/`.*

## 3. Process Data (NLP) / Collect Data
Process the raw text to generate embeddings and sentiment scores. Or collect fresh data asynchronously (fastest).

```bash
# Recommended for fresh collection
python main.py --step collect --async-collection --target 5000

# Recommended for processing already downloaded data
python main.py --step process
```
*Output: Generates `backend/data/processed/processed_news.json` (~156MB).*

## 4. Build Graph
Construct the temporal graph from the processed articles.

```bash
python main.py --step build
```
*Output: Saves graph structure and features to `backend/data/processed/`.*

## 5. Train Model
Train the Temporal Graph Network (TGN) model.

```bash
python main.py --step train --num-epochs 100
```
*Output: Saves the trained model to `backend/models/`.*

## 6. Run Benchmarks (Optional)
Compare the model against baselines.

```bash
python run_benchmarks.py
```
*Output: Saves report and plots to `backend/visualizations/`.*
