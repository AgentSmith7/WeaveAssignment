# Music Rights Processing Module

This module processes music royalty statement PDFs using the existing document-agent framework.

## Overview

The module handles:
1. **PDF to Image Conversion**: Converts multi-page PDFs to individual JPG images
2. **Page-by-Page Processing**: Processes each page through the existing pipeline (RT-DETR → OCR → LLM → Quality)
3. **Weave Instrumentation**: Full tracing and monitoring via Weave

## Folder Structure

```
music_rights/
├── data/
│   ├── input_pdfs/          # Place your PDF files here
│   ├── converted_images/    # Auto-generated page images
│   │   └── {pdf_name}/      # Subfolder per PDF
│   │       ├── page_001.jpg
│   │       ├── page_002.jpg
│   │       └── ...
│   └── output/              # Processing results (JSON)
│
├── src/
│   └── pdf_converter.py     # PDF → JPG conversion
│
├── scripts/
│   ├── convert_pdfs.py      # Batch convert PDFs
│   └── process_statements.py # Run processing pipeline
│
└── README.md
```

## Usage

### Step 1: Add PDFs

Copy your music rights PDF files to:
```
music_rights/data/input_pdfs/
```

### Step 2: Convert PDFs to Images

```bash
cd document-agent/music_rights/scripts
python convert_pdfs.py
```

Options:
- `--dpi 200` - Lower DPI for faster processing (default: 300)
- `--format png` - Use PNG instead of JPG

### Step 3: Process Statements

```bash
python process_statements.py
```

Options:
- `--pdf "Statement_Name"` - Process specific PDF only
- `--pages "1-10"` - Process only pages 1-10
- `--pages "5-5"` - Process single page

### Examples

```bash
# Convert all PDFs
python convert_pdfs.py

# Process all converted PDFs
python process_statements.py

# Process first 5 pages of a specific PDF
python process_statements.py --pdf "2023 12 - Statement_33877_023_40469_566473_20231231" --pages "1-5"

# Process only summary pages (typically pages 2-6)
python process_statements.py --pages "2-6"
```

## Output

Results are saved to `data/output/{pdf_name}_results.json` with:
- Per-page extraction results
- Detected regions with bounding boxes
- Extracted text and structured content
- Quality assessments
- Processing metadata

## Pipeline

Each page goes through the existing document-agent pipeline:

```
Page Image → RT-DETR Layout Detection → VLM OCR → LLM Content Analysis → Quality Assessment → Hallucination Check
```

All operations are traced in Weave for monitoring and debugging.

## Requirements

- Existing document-agent setup (see parent README)
- PyMuPDF (fitz) - for PDF conversion
- OpenAI API key - for VLM/LLM processing
- Weave/W&B credentials - for tracing
