"""
Document Processing Agent Demo App

Features:
- Live document processing with Weave traces
- Performance monitoring dashboard
- Batch processing capabilities
- Error debugging and optimization
- Scaling demonstrations
"""

import streamlit as st
import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import logging

# Weave imports
import weave
from weave.trace.api import get_current_call

# Document processing imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.document_agent import DocumentProcessingAgent
from src.evaluation.weave_instrumentation import (
    DocumentQualityScorer, LayoutDetectionAccuracyScorer, OCRPerformanceScorer,
    ContentAnalysisQualityScorer, DocumentProcessingPerformanceMonitor,
    DocumentProcessingErrorMonitor, DocumentQualityTrendMonitor,
    BatchProcessingMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Weave
def initialize_weave():
    """Initialize Weave with proper configuration"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize Weave
        wandb_entity = os.environ.get("WEAVE_ENTITY", "wandb-smle")
        wandb_project = os.environ.get("WEAVE_PROJECT", "document-processing-agent")
        
        client = weave.init(f"{wandb_entity}/{wandb_project}")
        logger.info(f"Weave initialized: {wandb_entity}/{wandb_project}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Weave: {e}")
        return None

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'document_agent' not in st.session_state:
        st.session_state.document_agent = None
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = DocumentProcessingPerformanceMonitor()
    if 'error_monitor' not in st.session_state:
        st.session_state.error_monitor = DocumentProcessingErrorMonitor()
    if 'quality_monitor' not in st.session_state:
        st.session_state.quality_monitor = DocumentQualityTrendMonitor()
    if 'batch_monitor' not in st.session_state:
        st.session_state.batch_monitor = BatchProcessingMonitor()
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []

# Create document processing agent
def create_document_agent():
    """Create and initialize the document processing agent"""
    try:
        # Define prompts
        content_analysis_prompt = weave.StringPrompt("""
        Analyze the following document regions and extract structured content.
        
        Document Type: {document_type}
        Detected Regions: {regions_json}
        
        Provide a JSON response with:
        - title: Document title
        - abstract: Summary or abstract
        - sections: List of sections with title and content
        - key_value_pairs: Key-value pairs from forms
        - tables: Structured table data
        - figures: Figure descriptions
        - metadata: Additional document metadata
        """)
        
        quality_assessment_prompt = weave.StringPrompt("""
        Assess the quality of document processing results.
        
        Content: {content}
        Regions: {regions}
        
        IMPORTANT: You must provide a JSON response with numeric scores between 0 and 1.
        
        Required JSON format:
        {{
            "overall_quality": 0.85,
            "clarity_score": 0.90,
            "completeness_score": 0.80,
            "issues": ["List any issues found"],
            "recommendations": ["List improvement recommendations"]
        }}
        
        Rate the quality from 0.0 (poor) to 1.0 (excellent).
        """)
        
        context_prompt = weave.StringPrompt("""
        You are processing a document with the following details:
        Document Path: {document_path}
        Processing Steps: {processing_steps}
        """)
        
        guardrail_prompt = weave.StringPrompt("""
        Check for hallucinations in the document processing results.
        Focus on factual accuracy and direct derivation from the source document.
        """)
        
        # Create agent
        agent = DocumentProcessingAgent(
            content_analysis_prompt=content_analysis_prompt,
            quality_assessment_prompt=quality_assessment_prompt,
            context_prompt=context_prompt,
            guardrail_prompt=guardrail_prompt
        )
        
        return agent
    except Exception as e:
        logger.error(f"Failed to create document agent: {e}")
        return None

# Main Streamlit app
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Document Processing Agent",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session_state()
    client = initialize_weave()
    
    # Header
    st.title("📄 Document Processing Agent Demo")
    st.markdown("## Weave Instrumentation Demo")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        layout_model = st.selectbox("Layout Detection Model", ["RT-DETR", "YOLO", "Custom"])
        ocr_engine = st.selectbox("OCR Engine", ["VLM (GPT-4V)", "pytesseract", "easyocr"])
        extraction_model = st.selectbox("Content Analysis Model", ["gpt-4o-mini", "gpt-4o", "claude-3"])
        quality_model = st.selectbox("Quality Assessment Model", ["gpt-4o-mini", "gpt-4o", "claude-3"])
        
        # Processing options
        st.subheader("Processing Options")
        document_type = st.selectbox("Document Type", ["auto", "academic_paper", "business_report", "form_document", "technical_manual"])
        batch_size = st.slider("Batch Size", 1, 10, 3)
        
        # Initialize agent button
        if st.button("🚀 Initialize Agent", type="primary"):
            with st.spinner("Initializing Document Processing Agent..."):
                st.session_state.document_agent = create_document_agent()
                if st.session_state.document_agent:
                    st.success("✅ Agent initialized successfully!")
                else:
                    st.error("❌ Failed to initialize agent")
    
    # Main content
    if st.session_state.document_agent is None:
        st.warning("⚠️ Please initialize the document processing agent first.")
        st.info("💡 Click 'Initialize Agent' in the sidebar to get started.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Single Document", "📚 Batch Processing", "📊 Monitoring", "🔍 Debugging", "⚡ Scaling"
    ])
    
    # Single Document Tab
    with tab1:
        st.header("📄 Single Document Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a document to process through the pipeline",
            accept_multiple_files=False
        )
        
        if uploaded_file:
            # Check file size (limit to 10MB)
            file_size = len(uploaded_file.getbuffer())
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                st.error("File too large. Please upload a file smaller than 10MB.")
                return
            
            # Save uploaded file
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File uploaded successfully: {uploaded_file.name} ({file_size:,} bytes)")
            
            # Process document
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing document through the pipeline..."):
                    start_time = time.time()
                    
                    try:
                        # Check if agent is initialized
                        if not hasattr(st.session_state, 'document_agent') or st.session_state.document_agent is None:
                            st.error("❌ Document agent not initialized. Please click 'Initialize Agent' first.")
                            return
                        
                        # Check if file exists
                        if not os.path.exists(file_path):
                            st.error(f"❌ File not found: {file_path}")
                            return
                        
                        # Process document
                        result = st.session_state.document_agent.predict(file_path, document_type)
                        
                        processing_time = time.time() - start_time
                        
                        # Check if result is valid
                        if not isinstance(result, dict):
                            st.error(f"❌ Processing failed: Invalid result type - {type(result)}")
                            st.session_state.error_monitor.track_error("processing_error", f"Invalid result type: {type(result)}", file_path)
                            return
                        
                        # Track performance
                        st.session_state.performance_monitor.track_processing_time(processing_time, document_type)
                        st.session_state.processing_history.append({
                            "timestamp": time.time(),
                            "document_path": file_path,
                            "document_type": document_type,
                            "processing_time": processing_time,
                            "success": "error" not in result
                        })
                        
                        if "error" in result:
                            st.error(f"❌ Processing failed: {result['error']}")
                            st.session_state.error_monitor.track_error("processing_error", result['error'], file_path)
                        else:
                            st.success("✅ Document processed successfully!")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Processing Time", f"{processing_time:.2f}s")
                            
                            with col2:
                                num_regions = result.get("processing_metadata", {}).get("num_regions_detected", 0)
                                st.metric("Regions Detected", num_regions)
                            
                            with col3:
                                # Extract quality score from nested structure
                                quality_assessment = result.get("quality_assessment", {})
                                quality = 0.0
                                
                                if "raw_assessment" in quality_assessment:
                                    raw_assessment = quality_assessment["raw_assessment"]
                                    # Check if it's a string (JSON) or dict
                                    if isinstance(raw_assessment, str):
                                        try:
                                            import json
                                            parsed_assessment = json.loads(raw_assessment)
                                            quality = parsed_assessment.get("overall_quality", 0.0)
                                        except json.JSONDecodeError:
                                            # Try regex extraction as fallback
                                            import re
                                            patterns = [
                                                r'"overall_quality":\s*([0-9.]+)',
                                                r'overall_quality[:\s]*([0-9.]+)',
                                                r'quality[:\s]*([0-9.]+)',
                                                r'score[:\s]*([0-9.]+)',
                                                r'([0-9.]+)\s*out of 1',
                                                r'([0-9.]+)\s*\/\s*1'
                                            ]
                                            
                                            for pattern in patterns:
                                                matches = re.findall(pattern, raw_assessment, re.IGNORECASE)
                                                if matches:
                                                    try:
                                                        quality = float(matches[0])
                                                        break
                                                    except ValueError:
                                                        continue
                                        except Exception:
                                            quality = 0.0
                                    elif isinstance(raw_assessment, dict):
                                        quality = raw_assessment.get("overall_quality", 0.0)
                                else:
                                    quality = quality_assessment.get("overall_quality", 0.0)
                                
                                st.metric("Quality Score", f"{quality:.2f}")
                            
                            # Show detailed results
                            st.subheader("📋 Processing Results")
                            
                            # Success message with key metrics
                            st.success(f"✅ Document processed successfully! Found {num_regions} regions with {quality:.2f} quality score.")
                            
                            # Bounding box visualization
                            if result.get("detected_regions"):
                                st.write("**🎯 Layout Detection Results:**")
                                
                                # Get document path for ground truth comparison
                                document_path = result.get("document_path", "")
                                if document_path and os.path.exists(document_path):
                                    try:
                                        # Load ground truth data
                                        from src.utils.document_processing import load_training_data
                                        try:
                                            ground_truth_data = load_training_data()
                                        except FileNotFoundError as e:
                                            st.warning(f"⚠️ Ground truth data not found: {e}")
                                            ground_truth_data = []
                                        
                                        # Find matching ground truth for this document
                                        doc_name = os.path.basename(document_path)
                                        doc_base = os.path.splitext(doc_name)[0]  # Remove extension
                                        
                                        # Remove temp_ prefix if present
                                        if doc_base.startswith('temp_'):
                                            doc_base = doc_base[5:]  # Remove 'temp_' prefix
                                        
                                        ground_truth = None
                                        for gt in ground_truth_data:
                                            gt_image_name = os.path.basename(gt['image_path'])
                                            gt_base = os.path.splitext(gt_image_name)[0]  # Remove extension
                                            if gt_base == doc_base:
                                                ground_truth = gt
                                                break
                                        
                                        if ground_truth and ground_truth_data:
                                            # Create bounding box visualization
                                            import cv2
                                            import numpy as np
                                            from PIL import Image
                                            
                                            # Load the image
                                            image = cv2.imread(document_path)
                                            if image is not None:
                                                # Create visualization
                                                vis_image = image.copy()
                                                
                                                # Color mapping for region types
                                                colors = {
                                                    'text': (0, 255, 0),      # Green
                                                    'title': (0, 0, 255),     # Red
                                                    'table': (255, 0, 0),     # Blue
                                                    'figure': (255, 255, 0),  # Cyan
                                                    'list': (255, 0, 255)     # Magenta
                                                }
                                                
                                                
                                                # Create separate images for comparison
                                                pred_image = image.copy()
                                                gt_image = image.copy()
                                                
                                                # Draw predicted bounding boxes only
                                                for region in result["detected_regions"]:
                                                    if isinstance(region, dict) and 'bbox' in region:
                                                        bbox = region['bbox']
                                                        region_type = region.get('region_type', 'unknown')
                                                        color = colors.get(region_type, (128, 128, 128))
                                                        
                                                        x1, y1, x2, y2 = map(int, bbox)
                                                        cv2.rectangle(pred_image, (x1, y1), (x2, y2), color, 3)
                                                        
                                                        # Add label
                                                        label = f"{region_type} ({region.get('confidence', 0):.2f})"
                                                        cv2.putText(pred_image, label, (x1, y1-10), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                
                                                # Draw ground truth bounding boxes only
                                                if 'regions' in ground_truth:
                                                    for region in ground_truth['regions']:
                                                        bbox = region.bbox
                                                        region_type = region.region_type
                                                        color = colors.get(region_type, (128, 128, 128))
                                                        
                                                        x1, y1, x2, y2 = map(int, bbox)
                                                        cv2.rectangle(gt_image, (x1, y1), (x2, y2), color, 3)
                                                        
                                                        # Add label with same color as predictions
                                                        cv2.putText(gt_image, f"GT-{region_type}", (x1, y1-10), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                
                                                # Convert to RGB for display
                                                pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
                                                gt_image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                                                
                                                # Display side-by-side comparison
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.image(pred_image_rgb, caption="Model Predictions", use_container_width=True)
                                                with col2:
                                                    st.image(gt_image_rgb, caption="Ground Truth", use_container_width=True)
                                                
                                                # Show legend
                                                st.write("**Legend:**")
                                                legend_cols = st.columns(len(colors))
                                                for i, (region_type, color) in enumerate(colors.items()):
                                                    with legend_cols[i]:
                                                        st.markdown(f"<span style='color:rgb({color[2]},{color[1]},{color[0]})'>■</span> {region_type.title()}", 
                                                                  unsafe_allow_html=True)
                                                
                                                st.write("**Note:** Left = Model Predictions, Right = Ground Truth")
                                                
                                            else:
                                                st.warning("Could not load image for visualization")
                                        else:
                                            st.info("No ground truth data found for this document")
                                            
                                    except Exception as e:
                                        st.warning(f"Could not create bounding box visualization: {str(e)}")
                                
                                # Show region counts
                                region_counts = {}
                                for i, region in enumerate(result["detected_regions"]):
                                    try:
                                        if isinstance(region, dict):
                                            region_type = region.get('region_type', 'unknown')
                                            region_counts[region_type] = region_counts.get(region_type, 0) + 1
                                    except Exception as e:
                                        st.warning(f"Error processing region: {str(e)}")
                                
                                if region_counts:
                                    st.write("**Region Counts:**")
                                    cols = st.columns(len(region_counts))
                                    for i, (region_type, count) in enumerate(region_counts.items()):
                                        with cols[i]:
                                            st.metric(f"{region_type.title()} Regions", count)
                                
                                st.write("---")
                            
                            
                            # Quality assessment
                            if result.get("quality_assessment"):
                                st.write("**Quality Assessment:**")
                                st.json(result["quality_assessment"])
                            
                            # Structured content
                            if result.get("structured_content"):
                                st.write("**Structured Content:**")
                                st.json(result["structured_content"])
                            
                            # Track quality
                            if "quality_assessment" in result:
                                quality_score = result["quality_assessment"].get("overall_quality", 0.0)
                                st.session_state.quality_monitor.track_quality(quality_score, document_type)
                    
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.session_state.error_monitor.track_error("exception", str(e), file_path)
                    
                    finally:
                        # Cleanup
                        if os.path.exists(file_path):
                            os.remove(file_path)
    
    # Batch Processing Tab
    with tab2:
        st.header("📚 Batch Processing")
        
        # Batch upload
        uploaded_files = st.file_uploader(
            "Upload Multiple Documents",
            type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple documents for batch processing"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} documents ready for processing")
            
            if st.button("🔄 Process Batch", type="primary"):
                # Save files
                file_paths = []
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = f"temp_batch_{i}_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Process batch
                with st.spinner(f"Processing {len(file_paths)} documents..."):
                    start_time = time.time()
                    
                    results = []
                    success_count = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file_path in enumerate(file_paths):
                        status_text.text(f"Processing document {i+1}/{len(file_paths)}")
                        
                        try:
                            result = st.session_state.document_agent.predict(file_path, document_type)
                            results.append(result)
                            
                            if "error" not in result:
                                success_count += 1
                            else:
                                st.session_state.error_monitor.track_error("batch_error", result["error"], file_path)
                        
                        except Exception as e:
                            results.append({"error": str(e)})
                            st.session_state.error_monitor.track_error("batch_exception", str(e), file_path)
                        
                        progress_bar.progress((i + 1) / len(file_paths))
                    
                    total_time = time.time() - start_time
                    
                    # Track batch performance
                    st.session_state.batch_monitor.track_batch(len(file_paths), success_count, total_time)
                    
                    # Display results
                    st.success(f"✅ Batch processing completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Documents", len(file_paths))
                    with col2:
                        st.metric("Successful", success_count)
                    with col3:
                        st.metric("Failed", len(file_paths) - success_count)
                    with col4:
                        st.metric("Success Rate", f"{success_count/len(file_paths)*100:.1f}%")
                    
                    # Show detailed results
                    st.subheader("📊 Batch Results")
                    results_df = pd.DataFrame([
                        {
                            "Document": os.path.basename(r.get("document_path", "unknown")),
                            "Success": "error" not in r,
                            "Processing Time": r.get("total_processing_time", 0.0),
                            "Quality Score": r.get("quality_assessment", {}).get("overall_quality", 0.0)
                        }
                        for r in results
                    ])
                    st.dataframe(results_df)
                    
                    # Cleanup
                    for file_path in file_paths:
                        if os.path.exists(file_path):
                            os.remove(file_path)
    
    # Monitoring Tab
    with tab3:
        st.header("📊 Performance Monitoring")
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        perf_summary = st.session_state.performance_monitor.get_performance_summary()
        if perf_summary:
            perf_df = pd.DataFrame([
                {
                    "Document Type": doc_type,
                    "Avg Time (s)": data["avg_processing_time"],
                    "Min Time (s)": data["min_processing_time"],
                    "Max Time (s)": data["max_processing_time"],
                    "Total Processed": data["total_processed"]
                }
                for doc_type, data in perf_summary.items()
            ])
            st.dataframe(perf_df)
            
            # Performance chart
            if len(perf_summary) > 1:
                fig = px.bar(
                    perf_df,
                    x="Document Type",
                    y="Avg Time (s)",
                    title="Average Processing Time by Document Type"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Error monitoring
        st.subheader("🚨 Error Monitoring")
        error_summary = st.session_state.error_monitor.get_error_summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Errors", error_summary.get("total_errors", 0))
        with col2:
            st.metric("Error Types", len(error_summary.get("error_types", {})))
        
        if error_summary.get("error_types"):
            error_df = pd.DataFrame([
                {"Error Type": error_type, "Count": count}
                for error_type, count in error_summary["error_types"].items()
            ])
            st.dataframe(error_df)
        
        # Quality trends
        st.subheader("📈 Quality Trends")
        quality_trends = st.session_state.quality_monitor.get_quality_trends()
        
        if quality_trends.get("trend") != "no_data":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend", quality_trends.get("trend", "unknown"))
            with col2:
                st.metric("Recent Avg", f"{quality_trends.get('recent_avg', 0):.2f}")
            with col3:
                st.metric("Overall Avg", f"{quality_trends.get('overall_avg', 0):.2f}")
        
        # Batch processing summary
        st.subheader("📚 Batch Processing Summary")
        batch_summary = st.session_state.batch_monitor.get_batch_summary()
        
        if batch_summary.get("status") != "no_batches_processed":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Batches", batch_summary.get("total_batches", 0))
            with col2:
                st.metric("Total Documents", batch_summary.get("total_documents", 0))
            with col3:
                st.metric("Success Rate", f"{batch_summary.get('overall_success_rate', 0)*100:.1f}%")
            with col4:
                st.metric("Avg Throughput", f"{batch_summary.get('avg_throughput', 0):.1f} docs/s")
    
    # Debugging Tab
    with tab4:
        st.header("🔍 Debugging & Optimization")
        
        st.subheader("🐛 Error Analysis")
        
        # Show recent errors
        error_summary = st.session_state.error_monitor.get_error_summary()
        if error_summary.get("recent_errors"):
            st.write("**Recent Errors:**")
            for error in error_summary["recent_errors"][-5:]:
                st.write(f"- **{error['error_type']}**: {error['error_message']}")
        
        st.subheader("⚡ Performance Optimization")
        
        # Processing history analysis
        if st.session_state.processing_history:
            history_df = pd.DataFrame(st.session_state.processing_history)
            
            # Performance by document type
            if len(history_df) > 1:
                perf_by_type = history_df.groupby('document_type')['processing_time'].agg(['mean', 'std', 'count'])
                st.write("**Performance by Document Type:**")
                st.dataframe(perf_by_type)
                
                # Performance trend chart
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='processing_time',
                    color='document_type',
                    title='Processing Time Trend'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔧 Optimization Recommendations")
        
        # Generate recommendations based on data
        recommendations = []
        
        if st.session_state.processing_history:
            avg_time = sum(h['processing_time'] for h in st.session_state.processing_history) / len(st.session_state.processing_history)
            if avg_time > 10:
                recommendations.append("⚠️ High processing times detected. Consider optimizing OCR settings or using faster models.")
        
        error_count = error_summary.get("total_errors", 0)
        if error_count > 0:
            recommendations.append(f"🚨 {error_count} errors detected. Review error logs and consider improving error handling.")
        
        if not recommendations:
            recommendations.append("✅ No optimization issues detected. System is performing well.")
        
        for rec in recommendations:
            st.write(rec)
    
    # Scaling Tab
    with tab5:
        st.header("⚡ Scaling & Production Considerations")
        
        st.subheader("📈 Current Performance")
        
        # Performance metrics
        perf_summary = st.session_state.performance_monitor.get_performance_summary()
        if perf_summary:
            total_processed = sum(data["total_processed"] for data in perf_summary.values())
            avg_time = sum(data["avg_processing_time"] for data in perf_summary.values()) / len(perf_summary)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents Processed", total_processed)
            with col2:
                st.metric("Average Processing Time", f"{avg_time:.2f}s")
            with col3:
                st.metric("Throughput", f"{1/avg_time:.2f} docs/s")
        
        st.subheader("🚀 Scaling Strategies")
        
        st.write("**1. Horizontal Scaling:**")
        st.write("- Deploy multiple agent instances behind a load balancer")
        st.write("- Use container orchestration (Kubernetes) for auto-scaling")
        st.write("- Implement queue-based processing for high throughput")
        
        st.write("**2. Model Optimization:**")
        st.write("- Use smaller, faster models for initial processing")
        st.write("- Implement model caching and warm-up strategies")
        st.write("- Consider model quantization for faster inference")
        
        st.write("**3. Infrastructure Optimization:**")
        st.write("- Use GPU acceleration for RT-DETR and OCR")
        st.write("- Implement document preprocessing pipelines")
        st.write("- Use CDN for document storage and retrieval")
        
        st.write("**4. Monitoring & Observability:**")
        st.write("- Set up real-time monitoring dashboards")
        st.write("- Implement alerting for performance degradation")
        st.write("- Use Weave for comprehensive tracing and debugging")
        
        st.subheader("📊 Production Metrics")
        
        # Simulate production metrics
        st.write("**Key Metrics to Monitor:**")
        metrics = [
            "Processing latency (P50, P95, P99)",
            "Throughput (documents per second)",
            "Error rate and error types",
            "Resource utilization (CPU, memory, GPU)",
            "Queue depth and processing backlog",
            "Model accuracy and quality scores"
        ]
        
        for metric in metrics:
            st.write(f"- {metric}")
        
        st.subheader("🔧 Production Deployment")
        
        st.write("**Deployment Architecture:**")
        st.code("""
        Load Balancer
            ↓
        API Gateway
            ↓
        Document Processing Service
            ↓
        RT-DETR Service → OCR Service → LLM Service
            ↓
        Weave Monitoring & Tracing
        """, language="text")
        
        st.write("**Configuration Management:**")
        st.write("- Environment-specific model configurations")
        st.write("- Dynamic model selection based on document type")
        st.write("- A/B testing for model performance")
        st.write("- Feature flags for gradual rollouts")

if __name__ == "__main__":
    main()
