"""
Weave Evaluation Implementation for Document Processing Pipeline
Adapted from the VLM NER blog recipe for document processing use case
"""

import json
import os
import time
import random
from typing import Dict, List, Any, Optional
import weave
from openai import OpenAI
import base64
from PIL import Image
import io
from dotenv import load_dotenv
from src.core.document_agent import DocumentProcessingAgent
from src.utils.document_processing import load_training_data

# Load environment variables
load_dotenv()

# Initialize Weave project
weave.init("document-processing-agent")

class DocumentProcessingEvaluation:
    """Weave Evaluation for Document Processing Pipeline"""
    
    def __init__(self):
        # Ensure API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.setup_prompts()
        self.setup_scorers()
        
        # Initialize our actual document processing agent
        self.document_agent = DocumentProcessingAgent(
            layout_model_path="models/best_model/best.pt",
            ocr_engine="vlm",
            extraction_model="gpt-4o",
            quality_model="gpt-4o", 
            guardrail_model="gpt-4o",
            content_analysis_prompt=weave.StringPrompt("Analyze the extracted content for structure and quality"),
            quality_assessment_prompt=weave.StringPrompt("Assess the quality of document processing"),
            context_prompt=weave.StringPrompt("Process this document with high accuracy"),
            guardrail_prompt=weave.StringPrompt("Check for hallucinations and errors")
        )
    
    def setup_prompts(self):
        """Create and publish prompts to Weave"""
        
        # Layout Detection Prompt
        layout_prompt = """
        Analyze this document image and identify all text regions, titles, tables, figures, and lists.
        Return a JSON array of regions with the following format:
        [
            {
                "region_type": "text|title|table|figure|list",
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95
            }
        ]
        """
        self.layout_prompt = weave.StringPrompt(layout_prompt)
        weave.publish(self.layout_prompt, name="layout-detection-prompt")
        
        # Content Extraction Prompt
        content_prompt = """
        Extract all text content from this document region.
        Return the text as it appears, maintaining formatting and structure.
        """
        self.content_prompt = weave.StringPrompt(content_prompt)
        weave.publish(self.content_prompt, name="content-extraction-prompt")
        
        # Quality Assessment Prompt
        quality_prompt = """
        Assess the quality of this document processing result.
        Return a JSON object with:
        {
            "overall_quality": 0.85,
            "clarity_score": 0.90,
            "completeness_score": 0.80,
            "issues": ["List any issues"],
            "recommendations": ["List improvements"]
        }
        """
        self.quality_prompt = weave.StringPrompt(quality_prompt)
        weave.publish(self.quality_prompt, name="quality-assessment-prompt")
    
    def setup_scorers(self):
        """Create Weave Scorers for evaluation"""
        
        @weave.op()
        def check_region_detection_accuracy(model_output, ground_truth):
            """Programmatic scorer for region detection accuracy with IoU matching"""
            if "error" in model_output:
                return False
            
            detected_regions = model_output.get("detected_regions", [])
            gt_regions = ground_truth.get("regions", [])
            
            if not detected_regions or not gt_regions:
                return False
            
            def calculate_iou(box1, box2):
                """Calculate Intersection over Union (IoU) between two bounding boxes"""
                x1_1, y1_1, x2_1, y2_1 = box1
                x1_2, y1_2, x2_2, y2_2 = box2
                
                # Calculate intersection
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)
                
                if x2_i <= x1_i or y2_i <= y1_i:
                    return 0.0
                
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                
                # Calculate union
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            
            def get_region_info(region):
                """Extract region information from different formats"""
                if hasattr(region, 'region_type'):
                    return region.region_type, region.bbox
                elif isinstance(region, dict):
                    return region.get('region_type', 'unknown'), region.get('bbox', [])
                else:
                    return 'unknown', []
            
            # Convert ground truth regions to standard format
            gt_standard = []
            for r in gt_regions:
                region_type, bbox = get_region_info(r)
                if bbox and len(bbox) == 4:
                    gt_standard.append({'region_type': region_type, 'bbox': bbox})
            
            # Convert detected regions to standard format
            detected_standard = []
            for r in detected_regions:
                region_type, bbox = get_region_info(r)
                if bbox and len(bbox) == 4:
                    detected_standard.append({'region_type': region_type, 'bbox': bbox})
            
            # Match regions using IoU threshold
            iou_threshold = 0.3  # Minimum IoU for a match
            matched_pairs = []
            used_gt_indices = set()
            used_det_indices = set()
            
            # Find best matches
            for i, det_region in enumerate(detected_standard):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_region in enumerate(gt_standard):
                    if j in used_gt_indices:
                        continue
                    
                    # Check if region types match
                    if det_region['region_type'] != gt_region['region_type']:
                        continue
                    
                    iou = calculate_iou(det_region['bbox'], gt_region['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_gt_idx != -1:
                    matched_pairs.append((i, best_gt_idx, best_iou))
                    used_gt_indices.add(best_gt_idx)
                    used_det_indices.add(i)
            
            # Calculate metrics
            true_positives = len(matched_pairs)
            false_positives = len(detected_standard) - true_positives
            false_negatives = len(gt_standard) - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Debug output
            print(f"Debug - Detected: {len(detected_standard)}, GT: {len(gt_standard)}")
            print(f"Debug - Matched pairs: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
            print(f"Debug - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
            
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "all_regions_detected": len(detected_standard) >= len(gt_standard)
            }
        
        @weave.op()
        def check_content_extraction_quality(model_output, ground_truth):
            """LLM-as-a-judge scorer for content extraction quality"""
            if "error" in model_output:
                return {"correct": False, "reason": "Processing error occurred"}
            
            extracted_content = model_output.get("structured_content", {})
            if not extracted_content:
                return {"correct": False, "reason": "No content extracted"}
            
            # Use LLM to evaluate content quality
            eval_prompt = """
            You are a document processing quality assessor. Evaluate the extracted content quality.
            
            Criteria:
            - Content should be complete and accurate
            - Text should be properly formatted
            - No missing or garbled text
            - Structure should be preserved
            
            Return JSON: {"correct": true/false, "reason": "explanation"}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": eval_prompt},
                        {"role": "user", "content": f"Evaluate this extracted content: {json.dumps(extracted_content, indent=2)}"}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            except Exception as e:
                return {"correct": False, "reason": f"Evaluation error: {str(e)}"}
        
        @weave.op()
        def check_processing_performance(model_output, ground_truth):
            """Performance scorer for processing efficiency"""
            if "error" in model_output:
                return {"performance_score": 0, "reason": "Processing failed"}
            
            processing_time = model_output.get("processing_metadata", {}).get("total_processing_time", 0)
            num_regions = model_output.get("processing_metadata", {}).get("num_regions_detected", 0)
            
            # Calculate performance metrics
            regions_per_second = num_regions / processing_time if processing_time > 0 else 0
            performance_score = min(1.0, regions_per_second / 10)  # Normalize to 0-1
            
            return {
                "performance_score": performance_score,
                "processing_time": processing_time,
                "regions_per_second": regions_per_second,
                "efficiency": "high" if performance_score > 0.7 else "medium" if performance_score > 0.4 else "low"
            }
        
        self.scorers = [
            check_region_detection_accuracy,
            check_content_extraction_quality,
            check_processing_performance
        ]
    
    def get_document_processing_pipeline(self):
        """Get the document processing pipeline function"""
        
        @weave.op()
        def document_processing_pipeline(image_base64, document_id):
            """Main document processing pipeline for evaluation"""
            try:
                print(f"Processing document: {document_id}")
                
                # Convert base64 to image
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                
                # Save image temporarily and use document processing agent
                temp_path = f"temp_{document_id}.png"
                image.save(temp_path)
                
                # Use our actual document processing agent
                result = self.document_agent.predict(
                    document_path=temp_path,
                    document_type="auto"
                )
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                print(f"Pipeline result for {document_id}: {result}")
                return result
                
            except Exception as e:
                print(f"Pipeline error for {document_id}: {e}")
                return {"error": str(e), "document_id": document_id}
        
        return document_processing_pipeline
    
    def create_evaluation_dataset(self, sample_documents, num_samples=10):
        """Create evaluation dataset from random sample documents"""
        # Pick random samples
        selected_docs = random.sample(sample_documents, min(num_samples, len(sample_documents)))
        print(f"Selected {len(selected_docs)} random documents for evaluation: {[os.path.basename(d) for d in selected_docs]}")
        
        dataset_rows = []
        
        for i, doc_path in enumerate(selected_docs):
            try:
                # Load and encode image
                with open(doc_path, "rb") as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode()
                
                # Get document ID from filename
                document_id = os.path.splitext(os.path.basename(doc_path))[0]
                
                # Load actual ground truth data
                ground_truth = self.load_ground_truth_data(document_id)
                
                dataset_rows.append({
                    "image_base64": image_base64,
                    "document_id": document_id,
                    "ground_truth": ground_truth
                })
                
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                continue
        
        return dataset_rows
    
    def load_ground_truth_data(self, document_id):
        """Load ground truth data for a document"""
        print(f"DEBUG: Loading ground truth for document_id: {document_id}")
        try:
            # Load training data to get ground truth
            training_data = load_training_data()
            print(f"DEBUG: Loaded {len(training_data)} training documents")
            
            # Find matching document by filename
            for doc in training_data:
                # Extract filename from image_path and compare with document_id
                image_path = doc.get('image_path', '')
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Remove temp_ prefix from document_id if present
                clean_document_id = document_id
                if document_id.startswith('temp_'):
                    clean_document_id = document_id[5:]  # Remove 'temp_' prefix
                
                print(f"Looking for {clean_document_id}, found {file_name}")
                
                if file_name == clean_document_id:
                    regions = doc.get('regions', [])
                    print(f"Found {len(regions)} ground truth regions for {document_id}")
                    return {
                        "regions": [
                            {
                                "region_type": region.region_type,
                                "bbox": region.bbox,
                                "confidence": 1.0  # Ground truth has 100% confidence
                            }
                            for region in regions
                        ]
                    }
            
            # If not found, return empty regions
            print(f"No ground truth found for document_id: {document_id}")
            print(f"Available documents: {[os.path.splitext(os.path.basename(doc.get('image_path', '')))[0] for doc in training_data[:5]]}")
            print(f"Total training data loaded: {len(training_data)} documents")
            return {"regions": []}
            
        except Exception as e:
            print(f"Error loading ground truth for {document_id}: {e}")
            return {"regions": []}
    
    async def run_evaluation(self, sample_documents, max_documents=10, evaluation_name="Document_Processing_Evaluation"):
        """Run the complete evaluation on random samples"""
        
        print(f"Testing evaluation on {max_documents} random documents from {len(sample_documents)} available")
        
        # Create dataset with random sampling
        dataset_rows = self.create_evaluation_dataset(sample_documents, num_samples=max_documents)
        
        # Create Weave dataset - use list directly
        dataset = dataset_rows
        
        # Create evaluation with custom name
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=self.scorers,
            name=evaluation_name
        )
        
        # Get the pipeline function
        pipeline_func = self.get_document_processing_pipeline()
        
        # Run evaluation
        results = await evaluation.evaluate(pipeline_func)
        
        return results

# Usage
async def main():
    """Run the evaluation"""
    
    # Initialize evaluation
    eval_system = DocumentProcessingEvaluation()
    
    # Get all available documents from the training data directory
    import glob
    sample_documents = glob.glob("data/samples/train/*.png")
    print(f"Found {len(sample_documents)} documents in training data")
    
    # Run evaluation on 10 random documents
    results = await eval_system.run_evaluation(sample_documents, max_documents=10)
    
    print("Evaluation completed!")
    print(f"Results: {results}")
    
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
