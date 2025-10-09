"""
Run Two Separate Document Processing Evaluations
This script runs two evaluations with different sets of 5 documents each
"""

import sys
import os
import asyncio
import random
import glob

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

from src.evaluation.weave_evaluation import DocumentProcessingEvaluation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_evaluation_batch(eval_system, sample_documents, batch_name, num_docs=5):
    """Run a single evaluation batch"""
    print(f"\n🚀 Starting {batch_name}...")
    print(f"📊 Running evaluation on {num_docs} random documents")
    
    # Run evaluation with custom name
    results = await eval_system.run_evaluation(
        sample_documents, 
        max_documents=num_docs,
        evaluation_name=batch_name
    )
    
    print(f"✅ {batch_name} completed!")
    print(f"📈 Results: {results}")
    
    return results

async def main():
    """Run two separate evaluations"""
    
    print("🔧 Initializing Document Processing Evaluation System...")
    
    # Initialize evaluation system
    eval_system = DocumentProcessingEvaluation()
    
    # Load ground truth data first to get available documents
    from src.utils.document_processing import load_training_data
    training_data = load_training_data()
    
    # Extract document paths from training data
    sample_documents = [doc['image_path'] for doc in training_data if 'image_path' in doc]
    print(f"📁 Found {len(sample_documents)} documents with ground truth data")
    
    if len(sample_documents) < 10:
        print(f"❌ Need at least 10 documents for two evaluations, found {len(sample_documents)}")
        return
    
    # Shuffle documents to ensure different selections
    random.shuffle(sample_documents)
    
    print("\n" + "="*60)
    print("🎯 EVALUATION BATCH 1: Academic Documents")
    print("="*60)
    
    # Run first evaluation
    results_1 = await run_evaluation_batch(
        eval_system, 
        sample_documents, 
        "Evaluation_Batch_1_Academic_Documents", 
        num_docs=5
    )
    
    print("\n" + "="*60)
    print("🎯 EVALUATION BATCH 2: Mixed Document Types")
    print("="*60)
    
    # Run second evaluation with different documents
    results_2 = await run_evaluation_batch(
        eval_system, 
        sample_documents, 
        "Evaluation_Batch_2_Mixed_Documents", 
        num_docs=5
    )
    
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    print(f"✅ Evaluation Batch 1 completed with {len(results_1) if results_1 else 0} results")
    print(f"✅ Evaluation Batch 2 completed with {len(results_2) if results_2 else 0} results")
    
    print(f"\n🌐 View your evaluations at: https://wandb.ai/rishabh29288/document-processing-agent/weave")
    print("📋 Both evaluations should now appear in your Weave dashboard!")
    
    return results_1, results_2

if __name__ == "__main__":
    asyncio.run(main())
