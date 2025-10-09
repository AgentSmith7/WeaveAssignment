"""
Weave Monitor Operations for Document Processing
These ops use ground truth data for real accuracy comparison
"""

import weave
import time
import logging
import os
import json
from typing import Dict, Any, List
from src.utils.document_processing import DocumentProcessor, load_training_data
from src.utils.extraction_rules import DocumentRegion

logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections_with_ground_truth(detected_regions, ground_truth_regions, iou_threshold=0.5):
    """Match detected regions with ground truth using IoU"""
    matches = []
    unmatched_detections = []
    unmatched_ground_truth = []
    
    # Create copies to avoid modifying originals
    detected_copy = detected_regions.copy()
    gt_copy = ground_truth_regions.copy()
    
    for detected in detected_copy:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_copy):
            if gt is None:  # Already matched
                continue
                
            iou = calculate_iou(detected.bbox, gt.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = i
        
        if best_gt_idx >= 0:
            matches.append({
                'detected': detected,
                'ground_truth': gt_copy[best_gt_idx],
                'iou': best_iou,
                'type_match': detected.region_type == gt_copy[best_gt_idx].region_type
            })
            gt_copy[best_gt_idx] = None  # Mark as matched
        else:
            unmatched_detections.append(detected)
    
    # Find unmatched ground truth
    for gt in gt_copy:
        if gt is not None:
            unmatched_ground_truth.append(gt)
    
    return matches, unmatched_detections, unmatched_ground_truth

@weave.op(name="document_quality_monitor")
def document_quality_monitor(
    document_path: str,
    document_type: str = "auto",
    quality_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Monitor document processing quality using real pipeline
    """
    start_time = time.time()
    
    try:
        # Process document with real pipeline
        processor = DocumentProcessor()
        result = processor.process_document(document_path, document_type)
        
        # Extract quality metrics
        quality_score = result.get("quality_assessment", {}).get("overall_quality", 0.0)
        num_regions = len(result.get("regions", []))
        processing_time = time.time() - start_time
        
        # Quality assessment
        quality_passed = quality_score >= quality_threshold
        regions_detected = num_regions > 0
        
        return {
            "quality_score": quality_score,
            "quality_passed": quality_passed,
            "num_regions_detected": num_regions,
            "regions_detected": regions_detected,
            "processing_time_seconds": processing_time,
            "document_type": document_type,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Document quality monitoring failed: {e}")
        return {
            "quality_score": 0.0,
            "quality_passed": False,
            "num_regions_detected": 0,
            "regions_detected": False,
            "processing_time_seconds": time.time() - start_time,
            "document_type": document_type,
            "success": False,
            "error": str(e)
        }

@weave.op(name="layout_detection_monitor")
def layout_detection_monitor(
    document_path: str,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Monitor layout detection accuracy using ground truth comparison
    """
    start_time = time.time()
    
    try:
        # Load ground truth data once and create lookup
        if not hasattr(layout_detection_monitor, '_ground_truth_cache'):
            layout_detection_monitor._ground_truth_cache = load_training_data()
            # Create filename lookup for efficiency
            layout_detection_monitor._ground_truth_lookup = {}
            for sample in layout_detection_monitor._ground_truth_cache:
                sample_filename = os.path.basename(sample['image_path'])
                sample_base = os.path.splitext(sample_filename)[0]
                layout_detection_monitor._ground_truth_lookup[sample_base] = sample['regions']
        
        # Find ground truth for this specific document
        doc_filename = os.path.basename(document_path)
        doc_base = os.path.splitext(doc_filename)[0]
        doc_ground_truth = layout_detection_monitor._ground_truth_lookup.get(doc_base, [])
        
        # Process document for layout detection
        processor = DocumentProcessor()
        document_images = processor._load_document(document_path)
        
        if not document_images:
            return {
                "success": False,
                "error": "No images loaded",
                "detected_regions": [],
                "num_regions": 0,
                "processing_time": time.time() - start_time
            }
        
        # Detect layout on first page
        detected_regions = processor.rtdetr_processor.detect_layout(document_images[0])
        
        # Match with ground truth if available
        if doc_ground_truth:
            matches, unmatched_detections, unmatched_ground_truth = match_detections_with_ground_truth(
                detected_regions, doc_ground_truth, iou_threshold
            )
            
            # Calculate accuracy metrics
            precision = len(matches) / len(detected_regions) if detected_regions else 0.0
            recall = len(matches) / len(doc_ground_truth) if doc_ground_truth else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate type accuracy
            type_matches = sum(1 for match in matches if match['type_match'])
            type_accuracy = type_matches / len(matches) if matches else 0.0
            
            # Calculate average IoU
            avg_iou = sum(match['iou'] for match in matches) / len(matches) if matches else 0.0
        else:
            # No ground truth available - just report detection stats
            matches = []
            unmatched_detections = detected_regions
            unmatched_ground_truth = []
            precision = recall = f1_score = type_accuracy = avg_iou = 0.0
        
        # Analyze results
        region_types = [r.region_type for r in detected_regions]
        unique_types = list(set(region_types))
        num_regions = len(detected_regions)
        
        return {
            "success": True,
            "detected_regions": region_types,
            "unique_region_types": unique_types,
            "num_regions": num_regions,
            "ground_truth_available": len(doc_ground_truth) > 0,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "type_accuracy": type_accuracy,
            "avg_iou": avg_iou,
            "num_matches": len(matches),
            "num_unmatched_detections": len(unmatched_detections),
            "num_unmatched_ground_truth": len(unmatched_ground_truth),
            "processing_time": time.time() - start_time,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Layout detection monitoring failed: {e}")
        return {
            "success": False,
            "detected_regions": [],
            "unique_region_types": [],
            "num_regions": 0,
            "ground_truth_available": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "type_accuracy": 0.0,
            "avg_iou": 0.0,
            "num_matches": 0,
            "num_unmatched_detections": 0,
            "num_unmatched_ground_truth": 0,
            "processing_time": time.time() - start_time,
            "error": str(e)
        }

@weave.op(name="content_extraction_monitor")
def content_extraction_monitor(
    document_path: str,
    region_type: str = "text"
) -> Dict[str, Any]:
    """
    Monitor content extraction quality for specific region types
    """
    start_time = time.time()
    
    try:
        # Process document with full pipeline
        processor = DocumentProcessor()
        result = processor.process_document(document_path, "auto")
        
        # Extract content metrics
        extracted_content = result.get("extracted_content", {})
        regions = result.get("regions", [])
        
        # Filter regions by type
        filtered_regions = [r for r in regions if r.get("region_type") == region_type]
        
        # Calculate content metrics
        total_content_length = sum(len(r.get("content", "")) for r in filtered_regions)
        avg_content_length = total_content_length / len(filtered_regions) if filtered_regions else 0
        content_extracted = total_content_length > 0
        
        # Calculate content quality metrics
        non_empty_content = sum(1 for r in filtered_regions if len(r.get("content", "").strip()) > 0)
        content_quality = non_empty_content / len(filtered_regions) if filtered_regions else 0.0
        
        return {
            "success": True,
            "region_type": region_type,
            "num_regions_of_type": len(filtered_regions),
            "total_content_length": total_content_length,
            "avg_content_length": avg_content_length,
            "content_extracted": content_extracted,
            "content_quality": content_quality,
            "non_empty_regions": non_empty_content,
            "processing_time": time.time() - start_time,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Content extraction monitoring failed: {e}")
        return {
            "success": False,
            "region_type": region_type,
            "num_regions_of_type": 0,
            "total_content_length": 0,
            "avg_content_length": 0.0,
            "content_extracted": False,
            "content_quality": 0.0,
            "non_empty_regions": 0,
            "processing_time": time.time() - start_time,
            "error": str(e)
        }

@weave.op(name="processing_performance_monitor")
def processing_performance_monitor(
    document_path: str,
    max_processing_time: float = 30.0
) -> Dict[str, Any]:
    """
    Monitor processing performance with real metrics
    """
    start_time = time.time()
    
    try:
        # Process document with full pipeline
        processor = DocumentProcessor()
        result = processor.process_document(document_path, "auto")
        
        processing_time = time.time() - start_time
        
        # Performance metrics
        performance_passed = processing_time <= max_processing_time
        quality_score = result.get("quality_assessment", {}).get("overall_quality", 0.0)
        num_regions = len(result.get("regions", []))
        
        # Calculate efficiency metrics
        regions_per_second = num_regions / processing_time if processing_time > 0 else 0
        quality_per_second = quality_score / processing_time if processing_time > 0 else 0
        
        # Memory and resource usage (if available)
        memory_usage = 0  # Could be enhanced to track actual memory usage
        
        return {
            "success": True,
            "processing_time_seconds": processing_time,
            "performance_passed": performance_passed,
            "quality_score": quality_score,
            "num_regions": num_regions,
            "regions_per_second": regions_per_second,
            "quality_per_second": quality_per_second,
            "memory_usage_mb": memory_usage,
            "efficiency_score": (regions_per_second * quality_per_second) if (regions_per_second > 0 and quality_per_second > 0) else 0.0,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        return {
            "success": False,
            "processing_time_seconds": time.time() - start_time,
            "performance_passed": False,
            "quality_score": 0.0,
            "num_regions": 0,
            "regions_per_second": 0.0,
            "quality_per_second": 0.0,
            "memory_usage_mb": 0,
            "efficiency_score": 0.0,
            "error": str(e)
        }
