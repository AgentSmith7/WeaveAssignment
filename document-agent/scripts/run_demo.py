#!/usr/bin/env python3
"""
Document Processing Agent Demo Launcher

This script launches the complete document processing demo with:
- Weave instrumentation
- Multi-agent workflow
- Performance monitoring
- Batch processing capabilities
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'streamlit', 'weave', 'langchain_openai', 'pandas', 
        'plotly', 'numpy', 'PIL', 'cv2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies found")
    return True

def check_environment():
    """Check environment variables and configuration"""
    logger.info("Checking environment...")
    
    # Check for .env file in parent directory
    env_file = Path("../.env")
    if not env_file.exists():
        logger.warning("⚠️ .env file not found. Creating template...")
        create_env_template()
        logger.info("📝 Please edit .env file with your API keys")
        return False
    
    # Check API keys
    from dotenv import load_dotenv
    load_dotenv("../.env")
    
    required_keys = ['OPENAI_API_KEY', 'WANDB_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
    
    if missing_keys:
        logger.error(f"Missing environment variables: {missing_keys}")
        logger.info("Please set these in your .env file")
        return False
    
    logger.info("✅ Environment configured")
    return True

def create_env_template():
    """Create .env template file"""
    env_content = """# Document Processing Agent Environment Variables

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Weights & Biases API Key
WANDB_API_KEY=your_wandb_api_key_here

# Weave Configuration
WEAVE_ENTITY=your_wandb_entity
WEAVE_PROJECT=document-processing-agent

# Optional: AWS credentials for Bedrock models
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AWS_SESSION_TOKEN=your_aws_session_token
"""
    
    with open("../.env", "w") as f:
        f.write(env_content)

def check_models():
    """Check if RT-DETR model is available"""
    logger.info("Checking models...")
    
    model_path = Path("../models/best_model/best.pt")
    if not model_path.exists():
        logger.warning("⚠️ RT-DETR model not found at models/best_model/best.pt")
        logger.info("The demo will use mock layout detection")
        return False
    
    logger.info("✅ RT-DETR model found")
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    logger.info("Checking Tesseract OCR...")
    
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ Tesseract OCR found")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.warning("⚠️ Tesseract OCR not found")
    logger.info("Install Tesseract OCR for full functionality")
    return False

def launch_demo():
    """Launch the Streamlit demo application"""
    logger.info("🚀 Launching Document Processing Agent Demo...")
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', '../src/ui/demo_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        logger.info("👋 Demo stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch demo: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🎯 Document Processing Agent Demo")
    print("=" * 60)
    
    # Check all requirements
    if not check_dependencies():
        print("❌ Dependency check failed")
        return 1
    
    if not check_environment():
        print("❌ Environment check failed")
        return 1
    
    check_models()
    check_tesseract()
    
    print("\n🚀 Starting demo application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the demo")
    print("=" * 60)
    
    # Launch the demo
    success = launch_demo()
    
    if success:
        print("✅ Demo completed successfully")
        return 0
    else:
        print("❌ Demo failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
