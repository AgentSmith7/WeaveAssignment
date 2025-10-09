# Document Processing Extraction Rules and Prompts
# This module contains prompts and rules for document processing pipeline

from typing import Dict, List, Any
from pydantic import BaseModel

# Document structure models
class DocumentRegion(BaseModel):
    """Represents a detected region in a document"""
    region_type: str  # text, table, figure, list, title
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    content: str
    page_number: int

class ExtractedContent(BaseModel):
    """Structured content extracted from a document"""
    title: str
    abstract: str
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    references: List[str]
    metadata: Dict[str, Any]

class DocumentQuality(BaseModel):
    """Document quality assessment"""
    overall_quality: float  # 0-1 scale
    clarity_score: float
    completeness_score: float
    structure_score: float
    issues: List[str]
    recommendations: List[str]

# Layout detection prompts
layout_detection_prompt = """
You are an expert document layout analyzer. Your task is to identify and classify different regions in a document.

For each detected region, provide:
1. Region type: text, table, figure, list, title, header, footer
2. Bounding box coordinates: [x1, y1, x2, y2] in pixels
3. Confidence score: 0.0 to 1.0
4. Brief description of the content

Document image: {document_image}

Analyze the document and return a JSON structure with detected regions.
"""

# OCR processing prompts
ocr_processing_prompt = """
You are an expert OCR text processor. Your task is to extract and clean text from document regions.

For each region, provide:
1. Raw extracted text
2. Cleaned and formatted text
3. Confidence score for the extraction
4. Any special formatting or structure detected

Region type: {region_type}
Region image: {region_image}
Bounding box: {bbox}

Extract and clean the text content.
"""

# Content analysis prompts
content_analysis_prompt = """
You are an expert document content analyzer. Your task is to understand and structure the content from different document regions.

Document type: {document_type}
Region type: {region_type}
Extracted text: {extracted_text}

Analyze the content and provide:
1. Main topics and themes
2. Key information and facts
3. Structure and organization
4. Important entities (names, dates, numbers, etc.)
5. Relationships between different pieces of information

Return a structured JSON response.
"""

# Academic paper extraction prompt
academic_paper_prompt = """
You are an expert academic paper analyzer. Extract structured information from this academic paper.

Paper content: {paper_content}

Extract the following information:
1. Title
2. Authors and affiliations
3. Abstract
4. Keywords
5. Introduction section
6. Methodology
7. Results and findings
8. Discussion and conclusions
9. References
10. Figures and tables with captions

Return a comprehensive JSON structure with all extracted information.
"""

# Form extraction prompt
form_extraction_prompt = """
You are an expert form analyzer. Extract structured information from this form document.

Form content: {form_content}

Extract the following information:
1. Form title and type
2. All form fields and their labels
3. Field types (text, checkbox, radio, dropdown, etc.)
4. Field values (if filled)
5. Required vs optional fields
6. Form sections and groupings
7. Instructions and help text
8. Submit buttons and actions

Return a structured JSON response with all form elements.
"""

# Report extraction prompt
report_extraction_prompt = """
You are an expert report analyzer. Extract structured information from this report document.

Report content: {report_content}

Extract the following information:
1. Report title and metadata
2. Executive summary
3. Table of contents structure
4. Main sections and subsections
5. Key findings and recommendations
6. Data tables and charts
7. Appendices and references
8. Report structure and organization

Return a comprehensive JSON structure with all extracted information.
"""

# Quality assessment prompt
quality_assessment_prompt = """
You are an expert document quality assessor. Evaluate the quality of document processing results.

Original document: {original_document}
Extracted content: {extracted_content}
Processing metadata: {processing_metadata}

Assess the following aspects:
1. Overall quality (0-1 scale)
2. Text extraction accuracy
3. Layout detection accuracy
4. Content completeness
5. Structural understanding
6. Any issues or errors
7. Recommendations for improvement

Return a detailed quality assessment with scores and explanations.
"""

# Context prompt for hallucination detection
context_prompt = """
You are analyzing a document processing result. The original document contained the following information:

Original document content: {original_content}
Extracted content: {extracted_content}
Processing steps: {processing_steps}

Use this context to evaluate whether the extracted content is accurate and based on the original document.
"""

# Guardrail prompt for hallucination detection
guardrail_prompt = """
You are a document processing quality controller. Your task is to detect hallucinations and inaccuracies in document processing results.

Original document: {original_document}
Processing result: {processing_result}

Check for:
1. Information that doesn't exist in the original document
2. Misinterpreted or incorrectly extracted content
3. Added information not present in the source
4. Structural errors in the extraction
5. Missing important information

Return a JSON response with:
- has_hallucination: boolean
- confidence: float (0-1)
- issues: list of specific problems found
- recommendations: list of improvements needed
"""

# Document comparison prompt
document_comparison_prompt = """
You are an expert document comparison analyzer. Compare two documents and identify similarities, differences, and relationships.

Document 1: {document1_content}
Document 2: {document2_content}

Analyze and provide:
1. Similarities between the documents
2. Key differences
3. Content overlap percentage
4. Structural differences
5. Quality comparison
6. Relationship between the documents
7. Recommendations for processing

Return a comprehensive comparison analysis.
"""

# Batch processing prompt
batch_processing_prompt = """
You are processing a batch of documents. Analyze the processing results and provide batch-level insights.

Documents processed: {num_documents}
Processing results: {processing_results}
Quality metrics: {quality_metrics}

Provide:
1. Overall batch processing quality
2. Common issues across documents
3. Performance metrics and statistics
4. Recommendations for improvement
5. Documents that need manual review
6. Batch processing summary

Return a comprehensive batch analysis report.
"""

# Expert review prompt
expert_review_prompt = """
You are an expert document processing reviewer. Review the processing results and provide expert feedback.

Document type: {document_type}
Processing result: {processing_result}
Quality metrics: {quality_metrics}

Provide expert review on:
1. Accuracy of content extraction
2. Quality of layout detection
3. Completeness of information
4. Structural understanding
5. Areas for improvement
6. Overall assessment

Return detailed expert feedback with specific recommendations.
"""

# Monitoring and metrics prompt
monitoring_prompt = """
You are a document processing system monitor. Analyze the processing metrics and system performance.

Processing metrics: {processing_metrics}
System performance: {system_performance}
Error logs: {error_logs}

Analyze and provide:
1. System health assessment
2. Performance trends
3. Error patterns and causes
4. Optimization recommendations
5. Alert conditions
6. Maintenance recommendations

Return a comprehensive monitoring report.
"""

# Document type classification prompt
document_classification_prompt = """
You are an expert document classifier. Classify the document type and provide relevant metadata.

Document content: {document_content}
Document structure: {document_structure}

Classify the document as one of:
- Academic paper
- Business report
- Form document
- Technical manual
- Legal document
- Financial report
- Research paper
- Other

Provide:
1. Document type classification
2. Confidence score
3. Key characteristics
4. Processing recommendations
5. Expected structure
6. Metadata extraction

Return a detailed classification result.
"""

# RT-DETR integration prompt
rt_detr_prompt = """
You are integrating RT-DETR layout detection results with document processing. 

RT-DETR results: {rt_detr_results}
Document image: {document_image}

Process the RT-DETR results and:
1. Validate detection accuracy
2. Refine bounding boxes if needed
3. Classify regions more precisely
4. Extract content from each region
5. Handle overlapping regions
6. Provide confidence scores

Return processed layout detection results with content extraction.
"""

# OCR quality assessment prompt
ocr_quality_prompt = """
You are an OCR quality assessor. Evaluate the quality of OCR text extraction.

Original image: {original_image}
OCR result: {ocr_result}
Region type: {region_type}

Assess:
1. Text extraction accuracy
2. Character recognition quality
3. Layout preservation
4. Formatting retention
5. Missing or misread text
6. Confidence in the extraction

Return a detailed OCR quality assessment.
"""

# Document assembly prompt
document_assembly_prompt = """
You are assembling a complete document from processed regions.

Document regions: {document_regions}
Extracted content: {extracted_content}
Document structure: {document_structure}

Assemble the document by:
1. Organizing regions in logical order
2. Maintaining document structure
3. Preserving formatting and layout
4. Ensuring content completeness
5. Creating a coherent document
6. Adding metadata and annotations

Return a complete assembled document with structure and content.
"""

# Export formats prompt
export_formats_prompt = """
You are preparing document processing results for export in different formats.

Processed document: {processed_document}
Export format: {export_format}
Requirements: {requirements}

Prepare the document for export by:
1. Formatting according to target format
2. Preserving structure and content
3. Adding necessary metadata
4. Ensuring compatibility
5. Optimizing for the target format
6. Including quality indicators

Return the formatted document ready for export.
"""

# All prompts dictionary for easy access
DOCUMENT_PROMPTS = {
    "layout_detection": layout_detection_prompt,
    "ocr_processing": ocr_processing_prompt,
    "content_analysis": content_analysis_prompt,
    "academic_paper": academic_paper_prompt,
    "form_extraction": form_extraction_prompt,
    "report_extraction": report_extraction_prompt,
    "quality_assessment": quality_assessment_prompt,
    "context": context_prompt,
    "guardrail": guardrail_prompt,
    "document_comparison": document_comparison_prompt,
    "batch_processing": batch_processing_prompt,
    "expert_review": expert_review_prompt,
    "monitoring": monitoring_prompt,
    "document_classification": document_classification_prompt,
    "rt_detr": rt_detr_prompt,
    "ocr_quality": ocr_quality_prompt,
    "document_assembly": document_assembly_prompt,
    "export_formats": export_formats_prompt
}

# Document type specific extraction rules
DOCUMENT_TYPE_RULES = {
    "academic_paper": {
        "required_sections": ["title", "abstract", "introduction", "methodology", "results", "conclusion", "references"],
        "optional_sections": ["acknowledgments", "appendix", "figures", "tables"],
        "quality_indicators": ["citations", "figures", "tables", "equations", "references"]
    },
    "business_report": {
        "required_sections": ["title", "executive_summary", "introduction", "findings", "recommendations"],
        "optional_sections": ["appendix", "charts", "tables", "references"],
        "quality_indicators": ["data_visualization", "metrics", "recommendations", "structure"]
    },
    "form_document": {
        "required_sections": ["form_title", "form_fields", "instructions"],
        "optional_sections": ["help_text", "validation_rules", "submit_actions"],
        "quality_indicators": ["field_completeness", "validation", "usability", "accessibility"]
    },
    "technical_manual": {
        "required_sections": ["title", "table_of_contents", "procedures", "troubleshooting"],
        "optional_sections": ["appendix", "glossary", "index", "diagrams"],
        "quality_indicators": ["step_clarity", "diagrams", "troubleshooting", "completeness"]
    }
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 0.9,
    "good": 0.8,
    "fair": 0.7,
    "poor": 0.6,
    "unacceptable": 0.5
}

# Processing pipeline stages
PROCESSING_STAGES = [
    "document_upload",
    "layout_detection",
    "region_classification", 
    "ocr_processing",
    "content_extraction",
    "quality_assessment",
    "expert_review",
    "final_assembly",
    "export"
]

# Error types and handling
ERROR_TYPES = {
    "layout_detection_error": "Failed to detect document layout",
    "ocr_error": "OCR processing failed",
    "content_extraction_error": "Content extraction failed",
    "quality_assessment_error": "Quality assessment failed",
    "expert_review_error": "Expert review failed",
    "assembly_error": "Document assembly failed",
    "export_error": "Export failed"
}

# Performance metrics
PERFORMANCE_METRICS = {
    "processing_time": "Time taken to process document",
    "accuracy_score": "Overall accuracy of processing",
    "layout_detection_accuracy": "Accuracy of layout detection",
    "ocr_accuracy": "Accuracy of OCR processing",
    "content_extraction_accuracy": "Accuracy of content extraction",
    "quality_score": "Overall quality score",
    "expert_review_score": "Expert review score"
}
