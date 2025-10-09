"""
Weave Instrumentation for Document Processing Agent
Custom evaluations, monitors, and scoring for document processing
"""

import weave
from typing import Dict, List, Any, Optional
import time
import json
import logging

logger = logging.getLogger(__name__)

# Custom Document Quality Scorer
@weave.op
class DocumentQualityScorer:
    """Custom scorer for document processing quality"""
    
    def score(self, model_output: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score document processing quality"""
        
        # Extract quality metrics
        quality_assessment = model_output.get("quality_assessment", {})
        processing_metadata = model_output.get("processing_metadata", {})
        
        # Calculate quality scores
        overall_quality = quality_assessment.get("overall_quality", 0.0)
        clarity_score = quality_assessment.get("clarity_score", 0.0)
        completeness_score = quality_assessment.get("completeness_score", 0.0)
        
        # Processing efficiency metrics
        num_regions = processing_metadata.get("num_regions_detected", 0)
        processing_time = processing_metadata.get("total_processing_time", 0.0)
        regions_per_second = num_regions / processing_time if processing_time > 0 else 0
        
        # OCR success metrics
        extraction_success_rate = processing_metadata.get("extraction_success_rate", 0.0)
        avg_text_per_region = processing_metadata.get("avg_text_per_region", 0.0)
        
        # Calculate composite quality score
        composite_score = (
            overall_quality * 0.4 +
            clarity_score * 0.3 +
            completeness_score * 0.3
        )
        
        return {
            "composite_quality_score": composite_score,
            "overall_quality": overall_quality,
            "clarity_score": clarity_score,
            "completeness_score": completeness_score,
            "processing_efficiency": regions_per_second,
            "ocr_success_rate": extraction_success_rate,
            "text_density": avg_text_per_region,
            "num_regions_detected": num_regions,
            "processing_time": processing_time,
            "quality_issues": len(quality_assessment.get("issues", [])),
            "quality_recommendations": len(quality_assessment.get("recommendations", []))
        }

# Layout Detection Accuracy Scorer
@weave.op
class LayoutDetectionAccuracyScorer:
    """Scorer for RT-DETR layout detection accuracy"""
    
    def score(self, model_output: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score layout detection accuracy"""
        
        detected_regions = model_output.get("detected_regions", [])
        processing_metadata = model_output.get("processing_metadata", {})
        
        # Calculate detection metrics
        num_regions = len(detected_regions)
        num_pages = processing_metadata.get("num_pages", 1)
        regions_per_page = num_regions / num_pages if num_pages > 0 else 0
        
        # Analyze region types
        region_types = [r.get("region_type", "unknown") for r in detected_regions]
        type_distribution = {}
        for region_type in region_types:
            type_distribution[region_type] = type_distribution.get(region_type, 0) + 1
        
        # Calculate confidence metrics
        confidences = [r.get("confidence", 0.0) for r in detected_regions if r.get("confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        high_confidence_regions = sum(1 for c in confidences if c > 0.8)
        
        return {
            "num_regions_detected": num_regions,
            "regions_per_page": regions_per_page,
            "avg_confidence": avg_confidence,
            "high_confidence_ratio": high_confidence_regions / len(confidences) if confidences else 0.0,
            "type_distribution": type_distribution,
            "detection_diversity": len(set(region_types)),
            "layout_detection_time": processing_metadata.get("layout_detection_time", 0.0)
        }

# OCR Performance Scorer
@weave.op
class OCRPerformanceScorer:
    """Scorer for OCR performance and accuracy"""
    
    def score(self, model_output: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score OCR performance"""
        
        processing_metadata = model_output.get("processing_metadata", {})
        extracted_text = model_output.get("extracted_text", "")
        
        # OCR metrics
        total_text_length = processing_metadata.get("total_text_length", 0)
        successful_extractions = processing_metadata.get("successful_extractions", 0)
        extraction_success_rate = processing_metadata.get("extraction_success_rate", 0.0)
        avg_text_per_region = processing_metadata.get("avg_text_per_region", 0.0)
        
        # Text quality metrics
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Processing efficiency
        ocr_processing_time = processing_metadata.get("ocr_processing_time", 0.0)
        text_per_second = total_text_length / ocr_processing_time if ocr_processing_time > 0 else 0
        
        return {
            "extraction_success_rate": extraction_success_rate,
            "total_text_length": total_text_length,
            "word_count": word_count,
            "avg_text_per_region": avg_text_per_region,
            "avg_word_length": avg_word_length,
            "text_per_second": text_per_second,
            "ocr_processing_time": ocr_processing_time,
            "successful_extractions": successful_extractions
        }

# Content Analysis Quality Scorer
@weave.op
class ContentAnalysisQualityScorer:
    """Scorer for LLM content analysis quality"""
    
    def score(self, model_output: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score content analysis quality"""
        
        structured_content = model_output.get("structured_content", {})
        processing_metadata = model_output.get("processing_metadata", {})
        
        # Content structure analysis
        content_keys = list(structured_content.keys()) if isinstance(structured_content, dict) else []
        content_complexity = len(content_keys)
        
        # Extract specific content elements
        title = structured_content.get("title", "")
        abstract = structured_content.get("abstract", "")
        sections = structured_content.get("sections", [])
        key_value_pairs = structured_content.get("key_value_pairs", {})
        
        # Calculate content richness metrics
        title_length = len(title) if title else 0
        abstract_length = len(abstract) if abstract else 0
        num_sections = len(sections) if isinstance(sections, list) else 0
        num_key_value_pairs = len(key_value_pairs) if isinstance(key_value_pairs, dict) else 0
        
        # Content completeness score
        completeness_indicators = [
            1 if title_length > 0 else 0,
            1 if abstract_length > 0 else 0,
            1 if num_sections > 0 else 0,
            1 if num_key_value_pairs > 0 else 0
        ]
        completeness_score = sum(completeness_indicators) / len(completeness_indicators)
        
        return {
            "content_complexity": content_complexity,
            "title_present": title_length > 0,
            "abstract_present": abstract_length > 0,
            "sections_count": num_sections,
            "key_value_pairs_count": num_key_value_pairs,
            "content_completeness": completeness_score,
            "content_analysis_time": processing_metadata.get("content_analysis_time", 0.0),
            "llm_model": processing_metadata.get("llm_model", "unknown")
        }

# Performance Monitor
@weave.op
class DocumentProcessingPerformanceMonitor:
    """Monitor for document processing performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def track_processing_time(self, processing_time: float, document_type: str):
        """Track processing time by document type"""
        if document_type not in self.metrics:
            self.metrics[document_type] = {"times": [], "count": 0}
        
        self.metrics[document_type]["times"].append(processing_time)
        self.metrics[document_type]["count"] += 1
        
        # Keep only last 100 measurements
        if len(self.metrics[document_type]["times"]) > 100:
            self.metrics[document_type]["times"] = self.metrics[document_type]["times"][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        for doc_type, data in self.metrics.items():
            times = data["times"]
            if times:
                summary[doc_type] = {
                    "avg_processing_time": sum(times) / len(times),
                    "min_processing_time": min(times),
                    "max_processing_time": max(times),
                    "total_processed": data["count"]
                }
        return summary

# Error Rate Monitor
@weave.op
class DocumentProcessingErrorMonitor:
    """Monitor for document processing errors"""
    
    def __init__(self):
        self.errors = []
        self.error_types = {}
    
    def track_error(self, error_type: str, error_message: str, document_path: str):
        """Track processing errors"""
        error_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": error_message,
            "document_path": document_path
        }
        self.errors.append(error_record)
        
        # Track error types
        if error_type not in self.error_types:
            self.error_types[error_type] = 0
        self.error_types[error_type] += 1
        
        # Keep only last 1000 errors
        if len(self.errors) > 1000:
            self.errors = self.errors[-1000:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        total_errors = len(self.errors)
        return {
            "total_errors": total_errors,
            "error_types": self.error_types,
            "recent_errors": self.errors[-10:] if self.errors else []
        }

# Quality Trend Monitor
@weave.op
class DocumentQualityTrendMonitor:
    """Monitor for document quality trends"""
    
    def __init__(self):
        self.quality_history = []
    
    def track_quality(self, quality_score: float, document_type: str, timestamp: float = None):
        """Track quality scores over time"""
        if timestamp is None:
            timestamp = time.time()
        
        self.quality_history.append({
            "timestamp": timestamp,
            "quality_score": quality_score,
            "document_type": document_type
        })
        
        # Keep only last 1000 measurements
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-1000:]
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trends"""
        if not self.quality_history:
            return {"trend": "no_data"}
        
        recent_scores = [q["quality_score"] for q in self.quality_history[-10:]]
        if len(recent_scores) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend
        avg_recent = sum(recent_scores) / len(recent_scores)
        avg_older = sum(self.quality_history[:-10]) / len(self.quality_history[:-10]) if len(self.quality_history) > 10 else avg_recent
        
        if avg_recent > avg_older * 1.05:
            trend = "improving"
        elif avg_recent < avg_older * 0.95:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_avg": avg_recent,
            "overall_avg": sum(q["quality_score"] for q in self.quality_history) / len(self.quality_history),
            "total_measurements": len(self.quality_history)
        }

# Batch Processing Monitor
@weave.op
class BatchProcessingMonitor:
    """Monitor for batch processing performance"""
    
    def __init__(self):
        self.batch_results = []
    
    def track_batch(self, batch_size: int, success_count: int, total_time: float):
        """Track batch processing results"""
        batch_record = {
            "timestamp": time.time(),
            "batch_size": batch_size,
            "success_count": success_count,
            "failure_count": batch_size - success_count,
            "total_time": total_time,
            "success_rate": success_count / batch_size if batch_size > 0 else 0,
            "throughput": batch_size / total_time if total_time > 0 else 0
        }
        self.batch_results.append(batch_record)
        
        # Keep only last 100 batches
        if len(self.batch_results) > 100:
            self.batch_results = self.batch_results[-100:]
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get batch processing summary"""
        if not self.batch_results:
            return {"status": "no_batches_processed"}
        
        total_batches = len(self.batch_results)
        total_documents = sum(b["batch_size"] for b in self.batch_results)
        total_successes = sum(b["success_count"] for b in self.batch_results)
        total_time = sum(b["total_time"] for b in self.batch_results)
        
        return {
            "total_batches": total_batches,
            "total_documents": total_documents,
            "overall_success_rate": total_successes / total_documents if total_documents > 0 else 0,
            "avg_throughput": total_documents / total_time if total_time > 0 else 0,
            "avg_batch_size": total_documents / total_batches if total_batches > 0 else 0
        }
