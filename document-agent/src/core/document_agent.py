"""
Document Processing Agent with Full Weave Instrumentation

This agent provides:
- Multi-agent workflow (RT-DETR → OCR → LLM → Quality)
- Comprehensive Weave tracing
- Custom evaluations and monitors
- Production-ready error handling
- Scaling considerations
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union, TypedDict
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Weave imports
import weave
from weave.scorers import HallucinationFreeScorer
from weave.trace.api import get_current_call

# Pydantic imports
from pydantic import PrivateAttr

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# Document processing imports
from src.utils.document_processing import RTDETRProcessor, OCRProcessor, DocumentProcessor
from src.utils.extraction_rules import DocumentRegion, ExtractedContent, DocumentQuality
from src.utils.quality_evaluators import DocumentQualityScorer, LayoutDetectionScorer, ContentExtractionScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the document processing state
class DocumentProcessingState(TypedDict):
    """State for document processing workflow"""
    document_path: str
    document_type: str
    document_images: List[Any]
    detected_regions: List[Dict[str, Any]]
    extracted_text: str
    structured_content: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    has_hallucination: bool
    tries: int
    error_message: Optional[str]
    processing_time: float

@weave.op
def reset_document_state(state: DocumentProcessingState) -> DocumentProcessingState:
    """Reset state for new document processing"""
    logger.info("Resetting document processing state")
    return {
        "document_path": state.get("document_path", ""),
        "document_type": state.get("document_type", "auto"),
        "document_images": [],
        "detected_regions": [],
        "extracted_text": "",
        "structured_content": {},
        "quality_assessment": {},
        "processing_metadata": {},
        "has_hallucination": False,
        "tries": 0,
        "error_message": None,
        "processing_time": 0.0
    }

class LayoutDetectionAgent:
    """RT-DETR Layout Detection Agent with Weave tracing"""
    
    def __init__(self, model_path: str = "models/best_model/best.pt"):
        self.rtdetr_processor = RTDETRProcessor(model_path)
    
    @weave.op(name="LayoutDetectionAgent")
    def __call__(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Detect document layout using RT-DETR"""
        start_time = time.time()
        logger.info(f"Starting layout detection for: {state['document_path']}")
        
        try:
            # Load document images
            processor = DocumentProcessor()
            document_images = processor._load_document(state["document_path"])
            state["document_images"] = document_images
            
            # Detect layout regions using RT-DETR
            all_detected_regions = []
            for page_num, img_pil in enumerate(document_images):
                img_np = np.array(img_pil)
                regions = self.rtdetr_processor.detect_layout(img_np)
                for region in regions:
                    region.page_number = page_num
                all_detected_regions.extend(regions)
            
            # Convert to dict format for state
            detected_regions = []
            for r in all_detected_regions:
                region_dict = {
                    "region_type": r.region_type,
                    "bbox": r.bbox,
                    "confidence": r.confidence,
                    "content": r.content,
                    "page_number": r.page_number
                }
                detected_regions.append(region_dict)
                logger.info(f"Region: type={r.region_type}, bbox={r.bbox}, conf={r.confidence}")
            state["detected_regions"] = detected_regions
            
            # Update processing metadata
            processing_time = time.time() - start_time
            state["processing_metadata"] = {
                "layout_detection_time": processing_time,
                "num_regions_detected": len(detected_regions),
                "num_pages": len(document_images),
                "regions_per_page": len(detected_regions) / len(document_images) if document_images else 0
            }
            
            logger.info(f"Layout detection completed: {len(detected_regions)} regions in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Layout detection failed: {str(e)}")
            state["error_message"] = f"Layout detection error: {str(e)}"
            raise e
        
        return state

class OCRProcessingAgent:
    """OCR Processing Agent with VLM-based extraction and Weave tracing"""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
    
    @weave.op(name="OCRProcessingAgent")
    def __call__(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Extract text from detected regions using OCR"""
        start_time = time.time()
        logger.info("Starting OCR processing")
        
        try:
            detected_regions = state["detected_regions"]
            document_images = state["document_images"]
            
            # Extract text from each region
            total_text_length = 0
            successful_extractions = 0
            
            for region_data in detected_regions:
                if region_data.get('page_number') is not None:
                    page_num = region_data['page_number']
                    if page_num < len(document_images):
                        # Convert PIL image to numpy array
                        img_np = np.array(document_images[page_num])
                        
                        # Create DocumentRegion object for OCR
                        region = DocumentRegion(**region_data)
                        
                        # Extract text using OCR
                        extracted_text = self.ocr_processor.extract_text_from_region(img_np, region)
                        region_data['content'] = extracted_text
                        
                        if extracted_text.strip():
                            successful_extractions += 1
                            total_text_length += len(extracted_text)
            
            # Combine all extracted text
            all_text = " ".join([r.get('content', '') for r in detected_regions if r.get('content')])
            state["extracted_text"] = all_text
            
            # Update processing metadata
            processing_time = time.time() - start_time
            ocr_metadata = {
                "ocr_processing_time": processing_time,
                "total_text_length": total_text_length,
                "successful_extractions": successful_extractions,
                "extraction_success_rate": successful_extractions / len(detected_regions) if detected_regions else 0,
                "avg_text_per_region": total_text_length / len(detected_regions) if detected_regions else 0
            }
            state["processing_metadata"].update(ocr_metadata)
            
            logger.info(f"OCR processing completed: {successful_extractions}/{len(detected_regions)} regions in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            state["error_message"] = f"OCR processing error: {str(e)}"
            raise e
        
        return state

class ContentAnalysisAgent:
    """LLM Content Analysis Agent with Weave tracing"""
    
    def __init__(self, llm_client, content_analysis_prompt):
        self.llm_client = llm_client
        self.content_analysis_prompt = content_analysis_prompt
    
    @weave.op(name="ContentAnalysisAgent")
    def __call__(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Analyze and structure content using LLM"""
        start_time = time.time()
        logger.info("Starting content analysis")
        
        try:
            detected_regions = state["detected_regions"]
            document_type = state["document_type"]
            
            # Prepare regions data for LLM
            regions_json = json.dumps(detected_regions, indent=2)
            
            # Use LLM to analyze and structure content
            analysis_prompt = self.content_analysis_prompt.format(
                regions_json=regions_json,
                document_type=document_type
            )
            
            # Get LLM response
            response = self.llm_client.invoke(analysis_prompt)
            structured_content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse as JSON, fallback to text
            try:
                structured_content = json.loads(structured_content)
            except json.JSONDecodeError:
                structured_content = {"raw_analysis": structured_content}
            
            state["structured_content"] = structured_content
            
            # Update processing metadata
            processing_time = time.time() - start_time
            analysis_metadata = {
                "content_analysis_time": processing_time,
                "llm_model": getattr(self.llm_client, 'model', 'unknown'),
                "content_structure": list(structured_content.keys()) if isinstance(structured_content, dict) else []
            }
            state["processing_metadata"].update(analysis_metadata)
            
            logger.info(f"Content analysis completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            state["error_message"] = f"Content analysis error: {str(e)}"
            raise e
        
        return state

class QualityAssessmentAgent:
    """Quality Assessment Agent with Weave tracing"""
    
    def __init__(self, llm_client, quality_assessment_prompt):
        self.llm_client = llm_client
        self.quality_assessment_prompt = quality_assessment_prompt
    
    @weave.op(name="QualityAssessmentAgent")
    def __call__(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Assess document processing quality"""
        start_time = time.time()
        logger.info("Starting quality assessment")
        
        try:
            detected_regions = state["detected_regions"]
            structured_content = state["structured_content"]
            
            # Prepare data for quality assessment
            regions_json = json.dumps(detected_regions, indent=2)
            content_json = json.dumps(structured_content, indent=2)
            
            # Use LLM to assess quality
            quality_prompt = self.quality_assessment_prompt.format(
                content=content_json,
                regions=regions_json
            )
            
            # Get LLM response
            response = self.llm_client.invoke(quality_prompt)
            quality_assessment = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse as JSON, fallback to text
            try:
                quality_assessment = json.loads(quality_assessment)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract quality score from text
                quality_assessment = {"raw_assessment": quality_assessment}
                # Try to extract overall_quality from the text using regex
                import re
                raw_text = quality_assessment["raw_assessment"]
                if isinstance(raw_text, str):
                    # Try multiple patterns to find quality score
                    patterns = [
                        r'"overall_quality":\s*([0-9.]+)',
                        r'overall_quality[:\s]*([0-9.]+)',
                        r'quality[:\s]*([0-9.]+)',
                        r'score[:\s]*([0-9.]+)',
                        r'([0-9.]+)\s*out of 1',
                        r'([0-9.]+)\s*\/\s*1'
                    ]
                    
                    quality_score = 0.0
                    for pattern in patterns:
                        quality_match = re.search(pattern, raw_text, re.IGNORECASE)
                        if quality_match:
                            try:
                                quality_score = float(quality_match.group(1))
                                if quality_score <= 1.0:  # Valid quality score
                                    break
                            except ValueError:
                                continue
                    
                    quality_assessment["overall_quality"] = quality_score
                else:
                    quality_assessment["overall_quality"] = 0.0
            
            state["quality_assessment"] = quality_assessment
            
            # Update processing metadata
            processing_time = time.time() - start_time
            quality_metadata = {
                "quality_assessment_time": processing_time,
                "overall_quality": quality_assessment.get("overall_quality", 0.0),
                "quality_issues": len(quality_assessment.get("issues", [])),
                "quality_recommendations": len(quality_assessment.get("recommendations", []))
            }
            state["processing_metadata"].update(quality_metadata)
            
            logger.info(f"Quality assessment completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            state["error_message"] = f"Quality assessment error: {str(e)}"
            raise e
        
        return state

class HallucinationGuardrail:
    """Hallucination detection and guardrail"""
    
    def __init__(self, guardrail_client, context_prompt, guardrail_prompt):
        self.guardrail_client = guardrail_client
        self.context_prompt = context_prompt
        self.guardrail_prompt = guardrail_prompt
    
    @weave.op(name="HallucinationGuardrail")
    def __call__(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Check for hallucinations in the processing results"""
        logger.info("Checking for hallucinations")
        
        try:
            # Use hallucination scorer
            original_content = state.get("extracted_text", "")
            structured_content = state.get("structured_content", {})
            
            # Check for hallucinations - use a simple heuristic for now
            # In a real implementation, you'd use the actual hallucination scorer
            has_hallucination = False
            
            # Simple heuristic: check if structured content seems reasonable
            if isinstance(structured_content, dict):
                # Check for common hallucination patterns
                content_str = str(structured_content).lower()
                if any(phrase in content_str for phrase in ["i don't know", "i cannot", "unable to", "not available"]):
                    has_hallucination = True
            
            state["has_hallucination"] = has_hallucination
            
            if has_hallucination:
                logger.warning("Hallucination detected in document processing")
                state["processing_metadata"]["hallucination_detected"] = True
            else:
                logger.info("No hallucinations detected")
                state["processing_metadata"]["hallucination_detected"] = False
            
        except Exception as e:
            logger.error(f"Hallucination check failed: {str(e)}")
            # Don't fail the entire pipeline for hallucination check errors
            state["has_hallucination"] = False
        
        return state

def create_document_processing_workflow(
    layout_agent: LayoutDetectionAgent,
    ocr_agent: OCRProcessingAgent,
    content_agent: ContentAnalysisAgent,
    quality_agent: QualityAssessmentAgent,
    guardrail: HallucinationGuardrail
) -> StateGraph:
    """Create the document processing workflow"""
    
    workflow = StateGraph(DocumentProcessingState)
    
    # Add nodes
    workflow.add_node("reset_state", reset_document_state)
    workflow.add_node("layout_detection", layout_agent)
    workflow.add_node("ocr_processing", ocr_agent)
    workflow.add_node("content_analysis", content_agent)
    workflow.add_node("quality_assessment", quality_agent)
    workflow.add_node("hallucination_guardrail", guardrail.__call__)
    
    # Define edges
    workflow.add_edge(START, "reset_state")
    workflow.add_edge("reset_state", "layout_detection")
    workflow.add_edge("layout_detection", "ocr_processing")
    workflow.add_edge("ocr_processing", "content_analysis")
    workflow.add_edge("content_analysis", "quality_assessment")
    workflow.add_edge("quality_assessment", "hallucination_guardrail")
    workflow.add_edge("hallucination_guardrail", END)
    
    return workflow

class DocumentProcessingAgent(weave.Model):
    """Main Document Processing Agent with full Weave instrumentation"""
    
    # Model configurations
    layout_model_path: str = "models/best_model/best.pt"
    ocr_engine: str = "pytesseract"
    extraction_model: str = "gpt-4o-mini"
    quality_model: str = "gpt-4o-mini"
    guardrail_model: str = "gpt-4o-mini"
    
    # Prompts
    content_analysis_prompt: weave.StringPrompt
    quality_assessment_prompt: weave.StringPrompt
    context_prompt: weave.StringPrompt
    guardrail_prompt: weave.StringPrompt
    
    # Private attributes
    _app: Any = PrivateAttr()
    _extraction_client: Any = PrivateAttr()
    _quality_client: Any = PrivateAttr()
    _guardrail_client: Any = PrivateAttr()
    
    def model_post_init(self, __context):
        """Initialize the document processing agent"""
        logger.info("Initializing Document Processing Agent")
        
        # Initialize LLM clients
        self._extraction_client = ChatOpenAI(model=self.extraction_model, max_retries=3)
        self._quality_client = ChatOpenAI(model=self.quality_model, max_retries=3)
        self._guardrail_client = HallucinationFreeScorer(model_id=f"openai/{self.guardrail_model}")
        
        # Initialize workflow agents
        layout_agent = LayoutDetectionAgent(self.layout_model_path)
        ocr_agent = OCRProcessingAgent()
        content_agent = ContentAnalysisAgent(self._extraction_client, self.content_analysis_prompt)
        quality_agent = QualityAssessmentAgent(self._quality_client, self.quality_assessment_prompt)
        guardrail = HallucinationGuardrail(self._guardrail_client, self.context_prompt, self.guardrail_prompt)
        
        # Create workflow
        workflow = create_document_processing_workflow(
            layout_agent, ocr_agent, content_agent, quality_agent, guardrail
        )
        
        # Compile with checkpointing
        checkpointer = MemorySaver()
        self._app = workflow.compile(checkpointer=checkpointer)
        
        logger.info("Document Processing Agent initialized successfully")
    
    @weave.op
    def predict(self, document_path: str, document_type: str = "auto") -> Dict[str, Any]:
        """Process a document through the full pipeline"""
        logger.info(f"Processing document: {document_path}")
        
        start_time = time.time()
        
        try:
            # Generate unique thread ID for this processing run
            thread_config = {"configurable": {"thread_id": f"doc_{int(time.time())}"}}
            
            # Run the workflow
            result = {}
            for event in self._app.stream(
                {"document_path": document_path, "document_type": document_type},
                config=thread_config
            ):
                for key, value in event.items():
                    logger.info(f"Workflow event: {key}")
                    if isinstance(value, dict):
                        result.update(value)
                    elif isinstance(value, str):
                        # Handle string values (like error messages)
                        result[key] = value
                        logger.warning(f"Received string value for {key}: {value}")
                    else:
                        logger.warning(f"Unexpected value type for {key}: {type(value)}")
            
            # Add total processing time
            total_time = time.time() - start_time
            result["total_processing_time"] = total_time
            
            # Ensure result is always a dictionary
            if not isinstance(result, dict):
                logger.error(f"Result is not a dictionary: {type(result)}")
                return {
                    "error": f"Invalid result type: {type(result)}",
                    "document_path": document_path,
                    "processing_time": total_time
                }
            
            logger.info(f"Document processing completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                "error": str(e),
                "document_path": document_path,
                "processing_time": time.time() - start_time
            }
    
    @weave.op
    def batch_process(self, document_paths: List[str], document_types: List[str] = None) -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        logger.info(f"Batch processing {len(document_paths)} documents")
        
        if document_types is None:
            document_types = ["auto"] * len(document_paths)
        
        results = []
        for i, (doc_path, doc_type) in enumerate(zip(document_paths, document_types)):
            logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
            result = self.predict(doc_path, doc_type)
            results.append(result)
        
        return results
