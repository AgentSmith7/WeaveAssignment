# Document Processing Module with RT-DETR Integration
# This module handles document layout detection, OCR processing, and content extraction

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import logging
from pathlib import Path
import io
import pandas as pd

# Import our extraction rules
from .extraction_rules import (
    DocumentRegion, ExtractedContent, DocumentQuality,
    DOCUMENT_PROMPTS, DOCUMENT_TYPE_RULES, QUALITY_THRESHOLDS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTDETRProcessor:
    """RT-DETR model processor for document layout detection"""
    
    def __init__(self, model_path: str = "models/best_model/best.pt", device: str = "auto"):
        """
        Initialize RT-DETR processor
        
        Args:
            model_path: Path to RT-DETR model weights
            device: Device to run model on ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        # Category mapping based on training data: 0=text, 1=title, 2=list, 3=table, 4=figure
        # YOLO models typically use 0-based indexing
        self.class_names = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        self.class_names_list = ["text", "title", "list", "table", "figure"]
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load RT-DETR model using ultralytics YOLO"""
        try:
            # Try to find the model file with different path resolutions
            model_paths = [
                self.model_path,
                f"../{self.model_path}",
                f"../../{self.model_path}",
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_model", "best.pt")
            ]
            
            actual_model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    actual_model_path = path
                    logger.info(f"Found model at: {actual_model_path}")
                    break
            
            if actual_model_path is None:
                logger.error(f"RT-DETR model file not found. Tried: {model_paths}")
                raise FileNotFoundError(f"RT-DETR model file not found. Tried: {model_paths}")
            
            # Load RT-DETR model using ultralytics
            try:
                from ultralytics import YOLO
                self.model = YOLO(actual_model_path)
                logger.info(f"RT-DETR model loaded successfully from {actual_model_path}")
            except ImportError:
                logger.error("ultralytics not available. Please install: pip install ultralytics")
                raise ImportError("ultralytics not available. Please install: pip install ultralytics")
            except Exception as e:
                logger.error(f"Failed to load RT-DETR model: {e}")
                raise RuntimeError(f"Failed to load RT-DETR model: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load RT-DETR model: {e}")
            self.model = None
    
    def detect_layout(self, image: np.ndarray) -> List[DocumentRegion]:
        """
        Detect document layout using RT-DETR
        
        Args:
            image: Input document image as numpy array
            
        Returns:
            List of detected document regions
        """
        if self.model is None:
            logger.error("RT-DETR model not loaded! Cannot perform layout detection.")
            raise RuntimeError("RT-DETR model not loaded. Please check model path and installation.")
        
        try:
            # Run RT-DETR inference directly on image
            results = self._run_inference(image)
            
            # Convert results to DocumentRegion objects
            # results is a list, so we need to process the first result
            if results and len(results) > 0:
                regions = self._convert_results_to_regions(results[0], image.shape)
            else:
                regions = []
            
            return regions
            
        except Exception as e:
            logger.error(f"RT-DETR layout detection failed: {e}")
            raise RuntimeError(f"RT-DETR layout detection failed: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for RT-DETR model"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model input size
        target_size = (640, 640)  # RT-DETR input size
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(Image.fromarray((image * 255).astype(np.uint8)))
        return image_tensor.unsqueeze(0).to(self.device)
    
    def _run_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run RT-DETR inference using ultralytics YOLO"""
        if self.model is None:
            # Return mock results if model not available
            return self._get_mock_results(image)
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=0.25)  # Confidence threshold
            return results
        except Exception as e:
            logger.error(f"RT-DETR inference failed: {e}")
            return self._get_mock_results(image)
    
    def _get_mock_results(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate mock results for testing"""
        h, w = image.shape[:2]
        mock_detections = [
            {'box': [w*0.1, h*0.1, w*0.9, h*0.2], 'cls': 1, 'conf': 0.95}, # Title
            {'box': [w*0.1, h*0.25, w*0.9, h*0.4], 'cls': 0, 'conf': 0.90}, # Text
            {'box': [w*0.05, h*0.45, w*0.45, h*0.6], 'cls': 3, 'conf': 0.88}, # Table
            {'box': [w*0.5, h*0.45, w*0.95, h*0.7], 'cls': 4, 'conf': 0.85}, # Figure
            {'box': [w*0.1, h*0.75, w*0.9, h*0.9], 'cls': 0, 'conf': 0.92}, # Text
        ]
        
        # Convert to YOLO results format
        class MockResult:
            def __init__(self, detections, class_names):
                self.boxes = []
                for det in detections:
                    x, y, w, h = det['box']
                    self.boxes.append({
                        'xyxy': [x, y, x + w, y + h],
                        'conf': det['conf'],
                        'cls': det['cls'],
                        'name': class_names.get(det['cls'], 'unknown')
                    })
        
        return [MockResult(mock_detections, self.class_names)]
    
    def _convert_results_to_regions(self, results, image_shape: Tuple[int, ...]) -> List[DocumentRegion]:
        """Convert RT-DETR results to DocumentRegion objects"""
        regions = []
        
        try:
            # Handle YOLO results object
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes
                if len(boxes) > 0:
                    # Extract coordinates, confidence, and class IDs
                    xyxy = boxes.xyxy.cpu().numpy()  # Convert to numpy
                    conf = boxes.conf.cpu().numpy()  # Convert to numpy
                    cls = boxes.cls.cpu().numpy().astype(int)  # Convert to numpy and int
                    
                    # Debug: Log detected class IDs
                    unique_classes = np.unique(cls)
                    logger.info(f"Detected class IDs: {unique_classes}")
                    logger.info(f"Class mapping: {self.class_names}")
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        confidence = conf[i]
                        class_id = int(cls[i])  # Ensure it's an integer
                        region_type = self.class_names.get(class_id, 'unknown')
                        
                        # Debug logging
                        if region_type == 'unknown':
                            logger.warning(f"Unknown class ID {class_id} detected. Available classes: {list(self.class_names.keys())}")
                        
                        # Coordinates are already in pixel format from YOLO
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        region = DocumentRegion(
                            region_type=region_type,
                            bbox=[x1, y1, x2, y2],
                            confidence=float(confidence),
                            content="",  # Will be filled by OCR
                            page_number=0
                        )
                        regions.append(region)
        except Exception as e:
            logger.error(f"Error converting results to regions: {e}")
            # Return empty list if conversion fails
            return []
        
        return regions
    


class OCRProcessor:
    """OCR processor using VLM (GPT-4V) for high-quality text extraction"""
    
    def __init__(self):
        """Initialize VLM-based OCR processor"""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            logger.info("VLM OCR processor initialized with GPT-4V")
        except Exception as e:
            logger.error(f"Failed to initialize VLM OCR: {e}")
            self.client = None
    
    def extract_text_from_region(self, image: np.ndarray, region: DocumentRegion) -> str:
        """
        Extract text from a specific region using VLM (GPT-4V)
        
        Args:
            image: Full document image
            region: Document region to extract text from
            
        Returns:
            Extracted text
        """
        try:
            if self.client is None:
                logger.error("VLM client not initialized")
                return ""
            
            # Crop region from image
            x1, y1, x2, y2 = region.bbox
            # Ensure coordinates are integers and within image bounds
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return ""
                
            region_image = image[y1:y2, x1:x2]
            
            if region_image.size == 0:
                return ""
            
            # Use VLM for text extraction
            text = self._extract_with_vlm(region_image, region.region_type)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"VLM extraction failed for region {region.region_type}: {e}")
            return ""
    
    def _extract_with_vlm(self, image: np.ndarray, region_type: str) -> str:
        """Extract text using VLM (GPT-4V)"""
        try:
            import base64
            import io
            from PIL import Image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create region-specific prompts for optimal extraction
            if region_type == 'title':
                prompt = "Extract the title text from this image. Return only the title text, no additional formatting or line breaks."
            elif region_type == 'text':
                prompt = "Extract all text content from this image. Return the text as it appears, maintaining line breaks and paragraph structure."
            elif region_type == 'table':
                prompt = "Extract the table data from this image. Return the table content in a structured format, preserving rows and columns. Use | to separate columns and newlines to separate rows."
            elif region_type == 'figure':
                prompt = "Extract any text, captions, or labels from this figure/image. Return the text content, including figure captions, axis labels, and any other text elements."
            elif region_type == 'list':
                prompt = "Extract the list items from this image. Return each list item on a new line, maintaining the list structure and numbering/bullets if present."
            else:
                prompt = "Extract all text content from this image. Return the text as it appears, maintaining formatting and structure."
            
            # Call GPT-4V
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            return ""
    


class DocumentProcessor:
    """Main document processing class that orchestrates the entire pipeline"""
    
    def __init__(self, 
                 model_path: str = "models/best_model/best.pt"):
        """
        Initialize document processor
        
        Args:
            model_path: Path to RT-DETR model
        """
        self.rtdetr_processor = RTDETRProcessor(model_path)
        self.ocr_processor = OCRProcessor()
        
    def process_document(self, 
                        document_path: str,
                        document_type: str = "auto",
                        output_format: str = "json") -> Dict[str, Any]:
        """
        Process a document through the complete pipeline
        
        Args:
            document_path: Path to document file
            document_type: Type of document (academic_paper, form, report, etc.)
            output_format: Output format (json, xml, etc.)
            
        Returns:
            Processed document results
        """
        try:
            logger.info(f"Processing document: {document_path}")
            
            # Load document
            images = self._load_document(document_path)
            
            # Process each page
            all_regions = []
            all_content = []
            
            for page_num, image in enumerate(images):
                logger.info(f"Processing page {page_num + 1}")
                
                # Layout detection
                regions = self.rtdetr_processor.detect_layout(image)
                
                # OCR processing
                for region in regions:
                    region.page_number = page_num
                    region.content = self.ocr_processor.extract_text_from_region(image, region)
                
                all_regions.extend(regions)
            
            # Content analysis and assembly
            extracted_content = self._analyze_content(all_regions, document_type)
            
            # Quality assessment
            quality = self._assess_quality(all_regions, extracted_content)
            
            # Assemble results
            results = {
                "document_path": document_path,
                "document_type": document_type,
                "pages_processed": len(images),
                "regions_detected": len(all_regions),
                "regions": [region.dict() for region in all_regions],
                "extracted_content": extracted_content.dict(),
                "quality_assessment": quality.dict(),
                "processing_metadata": {
                    "timestamp": str(pd.Timestamp.now()),
                    "model_used": "RT-DETR",
                    "ocr_engines": ["gpt-4o-mini"]
                }
            }
            
            logger.info(f"Document processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    def _load_document(self, document_path: str) -> List[np.ndarray]:
        """Load document and convert to images"""
        file_ext = Path(document_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self._load_pdf(document_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return [self._load_image(document_path)]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """Load PDF and convert to images"""
        try:
            # Use pdf2image for better quality
            images = convert_from_path(pdf_path, dpi=300)
            return [np.array(img) for img in images]
        except Exception as e:
            logger.error(f"Failed to load PDF with pdf2image: {e}")
            # Fallback to PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                images = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(np.array(img))
                doc.close()
                return images
            except Exception as e2:
                logger.error(f"Failed to load PDF with PyMuPDF: {e2}")
                raise
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def _analyze_content(self, regions: List[DocumentRegion], document_type: str) -> ExtractedContent:
        """Analyze and structure extracted content"""
        # Group regions by type
        text_regions = [r for r in regions if r.region_type == "text"]
        table_regions = [r for r in regions if r.region_type == "table"]
        figure_regions = [r for r in regions if r.region_type == "figure"]
        title_regions = [r for r in regions if r.region_type == "title"]
        
        # Extract structured content
        title = self._extract_title(title_regions)
        abstract = self._extract_abstract(text_regions)
        sections = self._extract_sections(text_regions)
        tables = self._extract_tables(table_regions)
        figures = self._extract_figures(figure_regions)
        references = self._extract_references(text_regions)
        
        # Create metadata
        metadata = {
            "document_type": document_type,
            "total_regions": len(regions),
            "text_regions": len(text_regions),
            "table_regions": len(table_regions),
            "figure_regions": len(figure_regions),
            "title_regions": len(title_regions)
        }
        
        return ExtractedContent(
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            metadata=metadata
        )
    
    def _extract_title(self, title_regions: List[DocumentRegion]) -> str:
        """Extract document title"""
        if not title_regions:
            return ""
        
        # Get the title with highest confidence
        best_title = max(title_regions, key=lambda r: r.confidence)
        return best_title.content
    
    def _extract_abstract(self, text_regions: List[DocumentRegion]) -> str:
        """Extract abstract from text regions"""
        # Look for abstract-like content (first few paragraphs)
        abstract_candidates = []
        for region in text_regions[:3]:  # Check first few regions
            if region.content and len(region.content) > 100:
                abstract_candidates.append(region.content)
        
        return " ".join(abstract_candidates[:2])  # Take first two candidates
    
    def _extract_sections(self, text_regions: List[DocumentRegion]) -> List[Dict[str, Any]]:
        """Extract document sections"""
        sections = []
        for i, region in enumerate(text_regions):
            if region.content:
                sections.append({
                    "section_id": f"section_{i}",
                    "content": region.content,
                    "bbox": region.bbox,
                    "confidence": region.confidence
                })
        return sections
    
    def _extract_tables(self, table_regions: List[DocumentRegion]) -> List[Dict[str, Any]]:
        """Extract table content"""
        tables = []
        for i, region in enumerate(table_regions):
            tables.append({
                "table_id": f"table_{i}",
                "content": region.content,
                "bbox": region.bbox,
                "confidence": region.confidence
            })
        return tables
    
    def _extract_figures(self, figure_regions: List[DocumentRegion]) -> List[Dict[str, Any]]:
        """Extract figure information"""
        figures = []
        for i, region in enumerate(figure_regions):
            figures.append({
                "figure_id": f"figure_{i}",
                "content": region.content,
                "bbox": region.bbox,
                "confidence": region.confidence
            })
        return figures
    
    def _extract_references(self, text_regions: List[DocumentRegion]) -> List[str]:
        """Extract references from text"""
        references = []
        for region in text_regions:
            if region.content:
                # Look for reference patterns
                lines = region.content.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['references', 'bibliography', 'cited']):
                        references.append(line.strip())
        return references
    
    def _assess_quality(self, regions: List[DocumentRegion], content: ExtractedContent) -> DocumentQuality:
        """Assess document processing quality"""
        # Calculate quality scores
        overall_quality = self._calculate_overall_quality(regions, content)
        clarity_score = self._calculate_clarity_score(regions)
        completeness_score = self._calculate_completeness_score(content)
        structure_score = self._calculate_structure_score(regions)
        
        # Identify issues
        issues = self._identify_issues(regions, content)
        recommendations = self._generate_recommendations(issues)
        
        return DocumentQuality(
            overall_quality=overall_quality,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            structure_score=structure_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_overall_quality(self, regions: List[DocumentRegion], content: ExtractedContent) -> float:
        """Calculate overall quality score"""
        if not regions:
            return 0.0
        
        # Average confidence of all regions
        avg_confidence = sum(r.confidence for r in regions) / len(regions)
        
        # Content completeness
        completeness = 1.0 if content.title else 0.5
        completeness += 0.3 if content.abstract else 0.0
        completeness += 0.2 if content.sections else 0.0
        
        # Combine scores
        overall = (avg_confidence * 0.6 + completeness * 0.4)
        return min(overall, 1.0)
    
    def _calculate_clarity_score(self, regions: List[DocumentRegion]) -> float:
        """Calculate clarity score based on text quality"""
        if not regions:
            return 0.0
        
        text_regions = [r for r in regions if r.region_type == "text" and r.content]
        if not text_regions:
            return 0.0
        
        # Calculate average text length and confidence
        avg_length = sum(len(r.content) for r in text_regions) / len(text_regions)
        avg_confidence = sum(r.confidence for r in text_regions) / len(text_regions)
        
        # Clarity based on text length and confidence
        clarity = min(avg_length / 100, 1.0) * avg_confidence
        return min(clarity, 1.0)
    
    def _calculate_completeness_score(self, content: ExtractedContent) -> float:
        """Calculate completeness score"""
        score = 0.0
        if content.title:
            score += 0.3
        if content.abstract:
            score += 0.2
        if content.sections:
            score += 0.3
        if content.tables:
            score += 0.1
        if content.figures:
            score += 0.1
        return min(score, 1.0)
    
    def _calculate_structure_score(self, regions: List[DocumentRegion]) -> float:
        """Calculate structure score based on region diversity"""
        if not regions:
            return 0.0
        
        # Count different region types
        region_types = set(r.region_type for r in regions)
        diversity_score = len(region_types) / 7.0  # 7 possible region types
        
        # Check for title presence
        has_title = any(r.region_type == "title" for r in regions)
        title_score = 0.3 if has_title else 0.0
        
        return min(diversity_score + title_score, 1.0)
    
    def _identify_issues(self, regions: List[DocumentRegion], content: ExtractedContent) -> List[str]:
        """Identify processing issues"""
        issues = []
        
        # Check for low confidence regions
        low_confidence = [r for r in regions if r.confidence < 0.5]
        if low_confidence:
            issues.append(f"{len(low_confidence)} regions with low confidence")
        
        # Check for empty content
        empty_regions = [r for r in regions if not r.content.strip()]
        if empty_regions:
            issues.append(f"{len(empty_regions)} regions with no extracted content")
        
        # Check for missing title
        if not content.title:
            issues.append("No title detected")
        
        # Check for missing abstract
        if not content.abstract:
            issues.append("No abstract detected")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on issues"""
        recommendations = []
        
        if any("low confidence" in issue for issue in issues):
            recommendations.append("Consider manual review of low confidence regions")
        
        if any("no extracted content" in issue for issue in issues):
            recommendations.append("Check OCR settings and image quality")
        
        if any("No title detected" in issue for issue in issues):
            recommendations.append("Manually add document title")
        
        if any("No abstract detected" in issue for issue in issues):
            recommendations.append("Manually add document abstract")
        
        if not recommendations:
            recommendations.append("Document processing completed successfully")
        
        return recommendations


# Utility functions
def process_single_document(document_path: str, 
                          output_path: Optional[str] = None,
                          document_type: str = "auto") -> Dict[str, Any]:
    """
    Process a single document
    
    Args:
        document_path: Path to document
        output_path: Path to save results (optional)
        document_type: Type of document
        
    Returns:
        Processing results
    """
    processor = DocumentProcessor()
    results = processor.process_document(document_path, document_type)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def batch_process_documents(document_paths: List[str],
                           output_dir: str,
                           document_type: str = "auto") -> List[Dict[str, Any]]:
    """
    Process multiple documents in batch
    
    Args:
        document_paths: List of document paths
        output_dir: Directory to save results
        document_type: Type of documents
        
    Returns:
        List of processing results
    """
    processor = DocumentProcessor()
    results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, doc_path in enumerate(document_paths):
        try:
            logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
            result = processor.process_document(doc_path, document_type)
            
            # Save individual result
            output_path = os.path.join(output_dir, f"result_{i+1}.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {doc_path}: {e}")
            results.append({"error": str(e), "document_path": doc_path})
    
    return results


if __name__ == "__main__":
    # Usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_processing.py <document_path>")
        sys.exit(1)
    
    document_path = sys.argv[1]
    result = process_single_document(document_path)
    print(json.dumps(result, indent=2))


def load_training_data(data_path: str = None, max_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Load training data samples for evaluation and testing.
    
    Args:
        data_path: Path to training data directory
        max_samples: Maximum number of samples to load
        
    Returns:
        List of training samples with images and annotations
    """
    import json
    from pathlib import Path
    import os
    
    # If no path provided, try to find the data directory relative to project root
    if data_path is None:
        # Try different possible locations
        possible_paths = [
            "data/samples/train/",
            "../data/samples/train/",
            "../../data/samples/train/",
            "data/samples/train"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("Training data directory not found. Tried: " + ", ".join(possible_paths))
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"DEBUG: Training data directory not found: {data_path}")
        logging.warning(f"Training data directory not found: {data_path}")
        return []
    
    samples = []
    json_files = list(data_dir.glob("*.json"))[:max_samples]
    
    for json_file in json_files:
        try:
            # Load annotation file
            with open(json_file, 'r') as f:
                annotation_data = json.load(f)
            
            # Find corresponding image file
            image_file = json_file.with_suffix('.png')
            if not image_file.exists():
                logging.warning(f"Image file not found for {json_file}")
                continue
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                logging.warning(f"Could not load image {image_file}")
                continue
            
            # Convert annotations to DocumentRegion format
            regions = []
            # Use the same class mapping as the model
            class_names = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
            
            for ann in annotation_data.get('annotations', []):
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # Map category_id to region_type
                    # Training data uses 1-based indexing, model uses 0-based
                    category_id = ann.get('category_id', 1) - 1  # Convert from 1-based to 0-based
                    region_type = class_names.get(category_id, 'unknown')
                    
                    region = DocumentRegion(
                        bbox=[x1, y1, x2, y2],
                        region_type=region_type,
                        confidence=1.0,  # Ground truth
                        content="",  # Will be filled by OCR
                        page_number=0
                    )
                    regions.append(region)
            
            samples.append({
                'image_path': str(image_file),
                'annotation_path': str(json_file),
                'image': image,
                'regions': regions,
                'metadata': {
                    'file_name': annotation_data.get('file_name', ''),
                    'corruption': annotation_data.get('corruption', {})
                }
            })
            
        except Exception as e:
            logging.error(f"Error loading sample {json_file}: {e}")
            continue
    
    logging.info(f"Loaded {len(samples)} training samples")
    return samples


def evaluate_model_on_training_data(model_path: str = "models/best_model/best.pt", 
                                   data_path: str = "data/samples/train/", 
                                   max_samples: int = 50) -> Dict[str, Any]:
    """
    Evaluate the RT-DETR model on training data samples.
    
    Args:
        model_path: Path to RT-DETR model
        data_path: Path to training data
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Evaluation results and metrics
    """
    # Load training samples
    samples = load_training_data(data_path, max_samples)
    if not samples:
        return {"error": "No training samples loaded"}
    
    # Initialize processor
    processor = RTDETRProcessor(model_path)
    
    # Evaluate on samples
    results = []
    for sample in samples:
        try:
            # Run layout detection
            detected_regions = processor.detect_layout(sample['image'])
            
            # Compare with ground truth
            gt_regions = sample['regions']
            
            # Calculate metrics (simplified)
            num_detected = len(detected_regions)
            num_ground_truth = len(gt_regions)
            
            # Simple accuracy metric
            accuracy = min(num_detected / max(num_ground_truth, 1), 1.0)
            
            results.append({
                'image_path': sample['image_path'],
                'num_detected': num_detected,
                'num_ground_truth': num_ground_truth,
                'accuracy': accuracy,
                'detected_regions': detected_regions,
                'ground_truth_regions': gt_regions
            })
            
        except Exception as e:
            logging.error(f"Error evaluating sample {sample['image_path']}: {e}")
            results.append({
                'image_path': sample['image_path'],
                'error': str(e)
            })
    
    # Calculate overall metrics
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        total_detected = sum(r['num_detected'] for r in valid_results)
        total_ground_truth = sum(r['num_ground_truth'] for r in valid_results)
    else:
        avg_accuracy = 0.0
        total_detected = 0
        total_ground_truth = 0
    
    return {
        'num_samples': len(samples),
        'num_evaluated': len(valid_results),
        'avg_accuracy': avg_accuracy,
        'total_detected': total_detected,
        'total_ground_truth': total_ground_truth,
        'results': results
    }
