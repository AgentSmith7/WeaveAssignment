"""
Generate traces for Weave Monitor setup using Ground Truth Data
This script runs the monitor ops with real ground truth comparison
"""

import sys
import os
import random
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

import weave
from src.monitoring.monitor_ops import (
    document_quality_monitor,
    layout_detection_monitor, 
    content_extraction_monitor,
    processing_performance_monitor
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Generate traces for monitor setup using ground truth data"""
    
    # Initialize Weave
    entity = os.getenv("WEAVE_ENTITY", "rishabh29288")
    project = os.getenv("WEAVE_PROJECT", "document-processing-agent")
    
    print(f"🔧 Initializing Weave: {entity}/{project}")
    weave.init(f"{entity}/{project}")
    
    # Sample documents to process (with ground truth available)
    sample_docs = [
        "../data/samples/train/doc_00000.png",
        "../data/samples/train/doc_00001.png", 
        "../data/samples/train/doc_00002.png",
        "../data/samples/train/doc_00003.png",
        "../data/samples/train/doc_00004.png"
    ]
    
    print("📊 Generating traces for Weave Monitors with Ground Truth Comparison...")
    print("🎯 Using real RT-DETR model + ground truth data for accurate monitoring")
    
    # Generate traces for each monitor
    for i, doc_path in enumerate(sample_docs):
        if not os.path.exists(doc_path):
            print(f"⚠️  Skipping {doc_path} - file not found")
            continue
            
        print(f"\n📄 Processing {doc_path} ({i+1}/{len(sample_docs)})")
        
        # 1. Document Quality Monitor - Real pipeline quality assessment
        try:
            print("  🔍 Running document quality monitor...")
            quality_result = document_quality_monitor(
                document_path=doc_path,
                document_type="academic",
                quality_threshold=0.7
            )
            print(f"    ✅ Quality score: {quality_result['quality_score']:.2f}, Passed: {quality_result['quality_passed']}")
        except Exception as e:
            print(f"    ❌ Quality monitor failed: {e}")
        
        # 2. Layout Detection Monitor - Ground truth comparison
        try:
            print("  🎯 Running layout detection monitor with ground truth...")
            layout_result = layout_detection_monitor(
                document_path=doc_path,
                iou_threshold=0.5
            )
            if layout_result['ground_truth_available']:
                print(f"    ✅ Precision: {layout_result['precision']:.2f}, Recall: {layout_result['recall']:.2f}, F1: {layout_result['f1_score']:.2f}")
                print(f"    📊 Detected {layout_result['num_regions']} regions, {layout_result['num_matches']} matches with ground truth")
            else:
                print(f"    ⚠️  No ground truth available - detected {layout_result['num_regions']} regions")
        except Exception as e:
            print(f"    ❌ Layout monitor failed: {e}")
        
        # 3. Content Extraction Monitor - Real content quality metrics
        for region_type in ["text", "title", "table"]:
            try:
                print(f"  📝 Running content extraction monitor for {region_type}...")
                content_result = content_extraction_monitor(
                    document_path=doc_path,
                    region_type=region_type
                )
                print(f"    ✅ {content_result['num_regions_of_type']} {region_type} regions, {content_result['total_content_length']} chars, Quality: {content_result['content_quality']:.2f}")
            except Exception as e:
                print(f"    ❌ Content extraction monitor failed: {e}")
        
        # 4. Performance Monitor - Real performance metrics
        try:
            print("  ⚡ Running performance monitor...")
            perf_result = processing_performance_monitor(
                document_path=doc_path,
                max_processing_time=30.0
            )
            print(f"    ✅ Time: {perf_result['processing_time_seconds']:.2f}s, Passed: {perf_result['performance_passed']}")
            print(f"    📈 Efficiency: {perf_result['regions_per_second']:.1f} regions/sec, Quality/sec: {perf_result['quality_per_second']:.3f}")
        except Exception as e:
            print(f"    ❌ Performance monitor failed: {e}")
    
    print(f"\n🎉 Generated traces for {len(sample_docs)} documents!")
    print(f"🌐 View your traces at: https://wandb.ai/{entity}/{project}/weave")
    print("\n📋 Next steps:")
    print("1. Go to the Weave UI and navigate to 'Monitors'")
    print("2. Click 'New Monitor' to create monitors for the ops above")
    print("3. Configure LLM-as-a-Judge scorers for each monitor")
    print("4. Monitors now use real ground truth data for accurate evaluation!")

if __name__ == "__main__":
    main()
