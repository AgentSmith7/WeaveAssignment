# Document Processing Agent with Weave Instrumentation

A document processing system that combines RT-DETR layout detection, VLM-based content extraction, and comprehensive Weave observability for enterprise document analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Weights & Biases API Key
- RT-DETR trained model (see Setup section)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AgentSmith7/WeaveAssignment.git
cd WeaveAssignment/document-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file in document-agent/ directory
OPENAI_API_KEY=your_openai_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
WEAVE_ENTITY=your_weave_entity
WEAVE_PROJECT=document-processing-agent
```

5. **Download RT-DETR model**
```bash
# Place your trained RT-DETR model at:
# models/best_model/best.pt
# (Model not included in repo due to size - see Model Setup section)
```

### Running the Demo

1. **Start the Streamlit application**
```bash
python scripts/run_demo.py
```

2. **Access the dashboard**
- Open http://localhost:8501 in your browser
- Upload a document image
- View processing results with bounding box visualization

## 🏗️ Architecture

### Multi-Agent Pipeline
```
Document Input → Layout Detection → OCR Processing → Content Analysis → Quality Assessment → Output
```

### Key Components
- **RT-DETR Model**: Real-time document layout detection
- **VLM Integration**: GPT-4V for content extraction
- **LangGraph Workflow**: State management and agent orchestration
- **Weave Instrumentation**: Tracing, evaluations, and monitoring

### Core Agents
1. **Layout Detection Agent**: RT-DETR model for region detection
2. **OCR Processing Agent**: VLM-based text extraction
3. **Content Analysis Agent**: LLM content structuring
4. **Quality Assessment Agent**: Quality scoring and validation
5. **Hallucination Guardrail**: Accuracy validation

## 📊 Weave Integration

### Tracing
- Full end-to-end trace visibility
- Step-by-step processing logs
- Performance metrics and timing

### Evaluations
- Region detection accuracy (IoU-based)
- Content extraction quality
- Processing performance metrics
- Ground truth comparison

### Monitors
- Document quality monitoring
- Layout detection accuracy
- Content extraction metrics
- Processing performance tracking

### Running Evaluations
```bash
# Run evaluation on sample documents
python scripts/run_two_evaluations.py
```

### Setting up Monitors
```bash
# Generate monitor traces
python scripts/generate_monitor_traces.py
```

## 🎯 Features

### Document Processing
- **Layout Detection**: RT-DETR model for accurate region identification
- **Content Extraction**: VLM-based text extraction with region-specific prompts
- **Quality Assessment**: LLM-powered quality scoring
- **Hallucination Detection**: Built-in accuracy validation

### User Interface
- **Single Document Processing**: Upload and process individual documents
- **Batch Processing**: Process multiple documents simultaneously
- **Visual Results**: Bounding box visualization with ground truth comparison
- **Real-time Monitoring**: Live processing metrics and quality scores

### Observability
- **Weave Tracing**: Complete processing pipeline visibility
- **Custom Evaluations**: Accuracy and quality metrics
- **Live Monitors**: Real-time performance tracking
- **Debugging Tools**: Error tracing and performance analysis

## 📁 Project Structure

```
document-agent/
├── src/
│   ├── core/                 # Core agent implementation
│   ├── ui/                   # Streamlit dashboard
│   ├── utils/                # Processing utilities
│   ├── evaluation/           # Weave evaluations
│   └── monitoring/           # Monitor operations
├── scripts/                  # Demo and evaluation scripts
├── data/
│   └── samples/train/        # Training data samples
├── models/                   # RT-DETR model (not included)
└── requirements.txt         # Python dependencies
```

## 🔧 Model Setup

### RT-DETR Model
The system requires a trained RT-DETR model for layout detection:

1. **Train your own model** using the provided notebooks:
   - `models/notebooks/Phase1_Foundation_Building.ipynb`
   - `models/notebooks/Phase2_model_logging_diagnostics.ipynb`
   - `models/notebooks/Phase_3_VLM_integration.ipynb`

2. **Place the model** at: `models/best_model/best.pt`

3. **Model requirements**:
   - Input: RGB images (any size)
   - Output: Bounding boxes, class predictions, confidence scores
   - Classes: text, title, list, table, figure

### Training Data
- Sample training data included in `data/samples/train/`
- Ground truth annotations in JSON format
- 10 sample documents with annotations provided

## 🚀 Production Deployment

### Environment Setup
```bash
# Production requirements
pip install -r requirements.txt

# Set production environment variables
export OPENAI_API_KEY=your_key
export WANDB_API_KEY=your_key
export WEAVE_ENTITY=your_entity
export WEAVE_PROJECT=your_project
```

### Scaling Considerations
- **Model Loading**: RT-DETR model cached for performance
- **API Rate Limits**: Built-in retry logic and error handling
- **Memory Management**: Efficient image processing pipeline
- **Monitoring**: Comprehensive Weave instrumentation

## 📈 Performance Metrics

### Accuracy Targets
- **Layout Detection**: >95% region detection accuracy
- **Content Extraction**: >90% text extraction quality
- **Processing Speed**: <2 seconds per document
- **Quality Score**: >0.8 average quality assessment

### Monitoring
- Real-time performance tracking
- Quality score monitoring
- Error rate tracking
- Processing time metrics

## 🛠️ Development

### Running Tests
```bash
# Test document processing pipeline
python -m pytest tests/

# Test individual components
python scripts/test_pipeline.py
```

### Adding New Features
1. **New Agents**: Add to `src/core/document_agent.py`
2. **New Evaluations**: Add to `src/evaluation/`
3. **New Monitors**: Add to `src/monitoring/`
4. **UI Updates**: Modify `src/ui/demo_app.py`

## 📚 Documentation

- **Code Walkthrough**: `CODE_WALKTHROUGH.md`
- **API Reference**: See docstrings in source code
- **Weave Integration**: See evaluation and monitoring modules


