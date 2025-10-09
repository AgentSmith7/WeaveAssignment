# Document Quality Evaluators
# This module contains evaluators for document processing quality assessment

import weave
from typing import Dict, Any, List, Optional
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import document processing models
from .extraction_rules import DocumentQuality, QUALITY_THRESHOLDS, PERFORMANCE_METRICS

class DocumentQualityScorer(weave.Scorer):
    """Scorer for overall document processing quality"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.client = ChatOpenAI(model=model_id)
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score document processing quality
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            Quality score and metrics
        """
        try:
            # Extract quality metrics from output
            quality_assessment = output.get("quality_assessment", {})
            detected_regions = output.get("detected_regions", [])
            extracted_content = output.get("extracted_content", {})
            
            # Calculate quality scores
            overall_quality = quality_assessment.get("overall_quality", 0.0)
            clarity_score = quality_assessment.get("clarity_score", 0.0)
            completeness_score = quality_assessment.get("completeness_score", 0.0)
            structure_score = quality_assessment.get("structure_score", 0.0)
            
            # Calculate additional metrics
            num_regions = len(detected_regions)
            avg_confidence = np.mean([r.get("confidence", 0) for r in detected_regions]) if detected_regions else 0
            
            # Content completeness metrics
            has_title = bool(extracted_content.get("title"))
            has_abstract = bool(extracted_content.get("abstract"))
            num_sections = len(extracted_content.get("sections", []))
            num_tables = len(extracted_content.get("tables", []))
            num_figures = len(extracted_content.get("figures", []))
            
            # Calculate composite score
            composite_score = (
                overall_quality * 0.4 +
                clarity_score * 0.2 +
                completeness_score * 0.2 +
                structure_score * 0.2
            )
            
            # Determine quality level
            if composite_score >= QUALITY_THRESHOLDS["excellent"]:
                quality_level = "excellent"
            elif composite_score >= QUALITY_THRESHOLDS["good"]:
                quality_level = "good"
            elif composite_score >= QUALITY_THRESHOLDS["fair"]:
                quality_level = "fair"
            elif composite_score >= QUALITY_THRESHOLDS["poor"]:
                quality_level = "poor"
            else:
                quality_level = "unacceptable"
            
            return {
                "overall_quality": overall_quality,
                "clarity_score": clarity_score,
                "completeness_score": completeness_score,
                "structure_score": structure_score,
                "composite_score": composite_score,
                "quality_level": quality_level,
                "num_regions": num_regions,
                "avg_confidence": avg_confidence,
                "has_title": has_title,
                "has_abstract": has_abstract,
                "num_sections": num_sections,
                "num_tables": num_tables,
                "num_figures": num_figures
            }
            
        except Exception as e:
            return {
                "overall_quality": 0.0,
                "clarity_score": 0.0,
                "completeness_score": 0.0,
                "structure_score": 0.0,
                "composite_score": 0.0,
                "quality_level": "error",
                "error": str(e)
            }


class LayoutDetectionScorer(weave.Scorer):
    """Scorer for layout detection accuracy"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.client = ChatOpenAI(model=model_id)
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score layout detection accuracy
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            Layout detection score and metrics
        """
        try:
            detected_regions = output.get("detected_regions", [])
            
            if not detected_regions:
                return {
                    "layout_accuracy": 0.0,
                    "num_regions": 0,
                    "avg_confidence": 0.0,
                    "region_types": [],
                    "coverage_score": 0.0
                }
            
            # Calculate basic metrics
            num_regions = len(detected_regions)
            avg_confidence = np.mean([r.get("confidence", 0) for r in detected_regions])
            
            # Analyze region types
            region_types = [r.get("region_type", "unknown") for r in detected_regions]
            unique_types = list(set(region_types))
            
            # Calculate coverage score (diversity of region types)
            expected_types = ["text", "title", "table", "figure", "list"]
            detected_expected = [t for t in unique_types if t in expected_types]
            coverage_score = len(detected_expected) / len(expected_types)
            
            # Calculate layout accuracy based on confidence and coverage
            layout_accuracy = (avg_confidence * 0.7 + coverage_score * 0.3)
            
            return {
                "layout_accuracy": layout_accuracy,
                "num_regions": num_regions,
                "avg_confidence": avg_confidence,
                "region_types": unique_types,
                "coverage_score": coverage_score,
                "detected_expected_types": detected_expected
            }
            
        except Exception as e:
            return {
                "layout_accuracy": 0.0,
                "num_regions": 0,
                "avg_confidence": 0.0,
                "region_types": [],
                "coverage_score": 0.0,
                "error": str(e)
            }


class ContentExtractionScorer(weave.Scorer):
    """Scorer for content extraction quality"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.client = ChatOpenAI(model=model_id)
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score content extraction quality
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            Content extraction score and metrics
        """
        try:
            extracted_content = output.get("extracted_content", {})
            detected_regions = output.get("detected_regions", [])
            
            # Extract content metrics
            title = extracted_content.get("title", "")
            abstract = extracted_content.get("abstract", "")
            sections = extracted_content.get("sections", [])
            tables = extracted_content.get("tables", [])
            figures = extracted_content.get("figures", [])
            references = extracted_content.get("references", [])
            
            # Calculate content completeness
            has_title = bool(title.strip())
            has_abstract = bool(abstract.strip())
            num_sections = len(sections)
            num_tables = len(tables)
            num_figures = len(figures)
            num_references = len(references)
            
            # Calculate content quality scores
            title_score = 1.0 if has_title else 0.0
            abstract_score = 1.0 if has_abstract else 0.0
            sections_score = min(num_sections / 5.0, 1.0)  # Normalize to 5 sections max
            tables_score = min(num_tables / 3.0, 1.0)  # Normalize to 3 tables max
            figures_score = min(num_figures / 3.0, 1.0)  # Normalize to 3 figures max
            
            # Calculate text quality metrics
            total_text_length = sum(len(s.get("content", "")) for s in sections)
            avg_section_length = total_text_length / max(num_sections, 1)
            
            # Calculate overall content extraction score
            content_score = (
                title_score * 0.2 +
                abstract_score * 0.2 +
                sections_score * 0.3 +
                tables_score * 0.15 +
                figures_score * 0.15
            )
            
            return {
                "content_score": content_score,
                "has_title": has_title,
                "has_abstract": has_abstract,
                "num_sections": num_sections,
                "num_tables": num_tables,
                "num_figures": num_figures,
                "num_references": num_references,
                "total_text_length": total_text_length,
                "avg_section_length": avg_section_length,
                "title_score": title_score,
                "abstract_score": abstract_score,
                "sections_score": sections_score,
                "tables_score": tables_score,
                "figures_score": figures_score
            }
            
        except Exception as e:
            return {
                "content_score": 0.0,
                "has_title": False,
                "has_abstract": False,
                "num_sections": 0,
                "num_tables": 0,
                "num_figures": 0,
                "num_references": 0,
                "error": str(e)
            }


class OCRAccuracyScorer(weave.Scorer):
    """Scorer for OCR accuracy and text extraction quality"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.client = ChatOpenAI(model=model_id)
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score OCR accuracy and text extraction quality
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            OCR accuracy score and metrics
        """
        try:
            detected_regions = output.get("detected_regions", [])
            
            if not detected_regions:
                return {
                    "ocr_accuracy": 0.0,
                    "text_extraction_score": 0.0,
                    "avg_text_length": 0.0,
                    "regions_with_text": 0,
                    "total_regions": 0
                }
            
            # Analyze text extraction from regions
            regions_with_text = 0
            total_text_length = 0
            text_quality_scores = []
            
            for region in detected_regions:
                content = region.get("content", "")
                if content.strip():
                    regions_with_text += 1
                    total_text_length += len(content)
                    
                    # Simple text quality heuristics
                    text_length = len(content)
                    has_punctuation = any(p in content for p in ['.', '!', '?', ',', ';', ':'])
                    has_numbers = any(c.isdigit() for c in content)
                    has_letters = any(c.isalpha() for c in content)
                    
                    # Calculate text quality score
                    quality_score = 0.0
                    if text_length > 10:  # Minimum length
                        quality_score += 0.3
                    if has_punctuation:  # Proper punctuation
                        quality_score += 0.2
                    if has_numbers:  # Contains numbers
                        quality_score += 0.1
                    if has_letters:  # Contains letters
                        quality_score += 0.2
                    if text_length > 50:  # Substantial content
                        quality_score += 0.2
                    
                    text_quality_scores.append(quality_score)
            
            # Calculate metrics
            total_regions = len(detected_regions)
            regions_with_text_ratio = regions_with_text / total_regions if total_regions > 0 else 0
            avg_text_length = total_text_length / max(regions_with_text, 1)
            avg_text_quality = np.mean(text_quality_scores) if text_quality_scores else 0
            
            # Calculate overall OCR accuracy
            ocr_accuracy = (regions_with_text_ratio * 0.6 + avg_text_quality * 0.4)
            
            return {
                "ocr_accuracy": ocr_accuracy,
                "text_extraction_score": regions_with_text_ratio,
                "avg_text_length": avg_text_length,
                "regions_with_text": regions_with_text,
                "total_regions": total_regions,
                "avg_text_quality": avg_text_quality,
                "total_text_length": total_text_length
            }
            
        except Exception as e:
            return {
                "ocr_accuracy": 0.0,
                "text_extraction_score": 0.0,
                "avg_text_length": 0.0,
                "regions_with_text": 0,
                "total_regions": 0,
                "error": str(e)
            }


class ProcessingTimeScorer(weave.Scorer):
    """Scorer for processing time and performance metrics"""
    
    def __init__(self):
        pass
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score processing time and performance
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            Processing time metrics
        """
        try:
            processing_metadata = output.get("processing_metadata", {})
            detected_regions = output.get("detected_regions", [])
            
            # Extract timing information
            timestamp = processing_metadata.get("timestamp", "")
            regions_processed = processing_metadata.get("regions_processed", 0)
            quality_score = processing_metadata.get("quality_score", 0.0)
            
            # Calculate processing efficiency
            num_regions = len(detected_regions)
            processing_efficiency = regions_processed / max(num_regions, 1) if num_regions > 0 else 0
            
            # Calculate performance score
            performance_score = (processing_efficiency * 0.5 + quality_score * 0.5)
            
            return {
                "processing_efficiency": processing_efficiency,
                "performance_score": performance_score,
                "regions_processed": regions_processed,
                "quality_score": quality_score,
                "timestamp": timestamp
            }
            
        except Exception as e:
            return {
                "processing_efficiency": 0.0,
                "performance_score": 0.0,
                "regions_processed": 0,
                "quality_score": 0.0,
                "error": str(e)
            }


class DocumentTypeScorer(weave.Scorer):
    """Scorer for document type classification accuracy"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.client = ChatOpenAI(model=model_id)
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score document type classification accuracy
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            Document type classification metrics
        """
        try:
            processing_metadata = output.get("processing_metadata", {})
            extracted_content = output.get("extracted_content", {})
            
            # Extract document type information
            document_type = processing_metadata.get("document_type", "unknown")
            title = extracted_content.get("title", "")
            abstract = extracted_content.get("abstract", "")
            sections = extracted_content.get("sections", [])
            tables = extracted_content.get("tables", [])
            figures = extracted_content.get("figures", [])
            
            # Analyze document characteristics
            has_title = bool(title.strip())
            has_abstract = bool(abstract.strip())
            num_sections = len(sections)
            num_tables = len(tables)
            num_figures = len(figures)
            
            # Calculate document type confidence based on characteristics
            type_confidence = 0.0
            
            if document_type == "academic_paper":
                if has_title and has_abstract and num_sections > 3:
                    type_confidence = 0.9
                elif has_title and (has_abstract or num_sections > 2):
                    type_confidence = 0.7
                else:
                    type_confidence = 0.5
            elif document_type == "business_report":
                if has_title and num_sections > 2 and (num_tables > 0 or num_figures > 0):
                    type_confidence = 0.9
                elif has_title and num_sections > 1:
                    type_confidence = 0.7
                else:
                    type_confidence = 0.5
            elif document_type == "form_document":
                if num_tables > 0 and num_sections > 0:
                    type_confidence = 0.8
                elif num_tables > 0:
                    type_confidence = 0.6
                else:
                    type_confidence = 0.4
            else:
                type_confidence = 0.5
            
            return {
                "document_type": document_type,
                "type_confidence": type_confidence,
                "has_title": has_title,
                "has_abstract": has_abstract,
                "num_sections": num_sections,
                "num_tables": num_tables,
                "num_figures": num_figures,
                "classification_accuracy": type_confidence
            }
            
        except Exception as e:
            return {
                "document_type": "unknown",
                "type_confidence": 0.0,
                "classification_accuracy": 0.0,
                "error": str(e)
            }


# Composite scorer that combines all quality metrics
class CompositeDocumentScorer(weave.Scorer):
    """Composite scorer that combines all document processing quality metrics"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.quality_scorer = DocumentQualityScorer(model_id)
        self.layout_scorer = LayoutDetectionScorer(model_id)
        self.content_scorer = ContentExtractionScorer(model_id)
        self.ocr_scorer = OCRAccuracyScorer(model_id)
        self.time_scorer = ProcessingTimeScorer()
        self.type_scorer = DocumentTypeScorer(model_id)
    
    @weave.op
    def score(self, output: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score document processing using all quality metrics
        
        Args:
            output: Document processing output
            target: Ground truth target (optional)
            
        Returns:
            Comprehensive quality assessment
        """
        try:
            # Get scores from all individual scorers
            quality_score = self.quality_scorer.score(output, target)
            layout_score = self.layout_scorer.score(output, target)
            content_score = self.content_scorer.score(output, target)
            ocr_score = self.ocr_scorer.score(output, target)
            time_score = self.time_scorer.score(output, target)
            type_score = self.type_scorer.score(output, target)
            
            # Calculate composite score
            composite_score = (
                quality_score.get("composite_score", 0) * 0.3 +
                layout_score.get("layout_accuracy", 0) * 0.2 +
                content_score.get("content_score", 0) * 0.2 +
                ocr_score.get("ocr_accuracy", 0) * 0.15 +
                time_score.get("performance_score", 0) * 0.1 +
                type_score.get("classification_accuracy", 0) * 0.05
            )
            
            # Determine overall quality level
            if composite_score >= QUALITY_THRESHOLDS["excellent"]:
                overall_quality = "excellent"
            elif composite_score >= QUALITY_THRESHOLDS["good"]:
                overall_quality = "good"
            elif composite_score >= QUALITY_THRESHOLDS["fair"]:
                overall_quality = "fair"
            elif composite_score >= QUALITY_THRESHOLDS["poor"]:
                overall_quality = "poor"
            else:
                overall_quality = "unacceptable"
            
            return {
                "composite_score": composite_score,
                "overall_quality": overall_quality,
                "quality_metrics": quality_score,
                "layout_metrics": layout_score,
                "content_metrics": content_score,
                "ocr_metrics": ocr_score,
                "performance_metrics": time_score,
                "type_metrics": type_score
            }
            
        except Exception as e:
            return {
                "composite_score": 0.0,
                "overall_quality": "error",
                "error": str(e)
            }


# Export all scorers for easy import
__all__ = [
    "DocumentQualityScorer",
    "LayoutDetectionScorer", 
    "ContentExtractionScorer",
    "OCRAccuracyScorer",
    "ProcessingTimeScorer",
    "DocumentTypeScorer",
    "CompositeDocumentScorer"
]
