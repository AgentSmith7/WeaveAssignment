# Scripts Directory

This directory contains the main scripts for running the document processing agent.

## Available Scripts

### Core Scripts
- **`run_demo.py`** - Main Streamlit application
- **`run_two_evaluations.py`** - Run two separate evaluations for comparison
- **`generate_monitor_traces.py`** - Generate sample traces for Weave monitors

## Usage

### Running the Application
```bash
python run_demo.py
```

### Running Evaluations
```bash
python run_two_evaluations.py
```

### Generating Monitor Traces
```bash
python generate_monitor_traces.py
```

## Requirements

Make sure you have:
1. Python environment activated
2. `.env` file with API keys configured
3. RT-DETR model in `../models/best_model/best.pt`
4. Training data in `../data/samples/train/`
