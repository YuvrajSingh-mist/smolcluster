#!/usr/bin/env python3
"""
Download GPT-2 model weights and convert to safetensors format.
Run this script to pre-download model weights before distributed training.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from smolcluster.utils.model_downloader import download_and_convert_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model safetensors files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model identifier from model_weights.yaml (default: gpt2)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: src/data)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: from config)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models from config"
    )
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        from smolcluster.utils.model_downloader import load_model_weights_config
        config = load_model_weights_config()
        logger.info("Available models:")
        for name, info in config['models'].items():
            logger.info(f"  - {name}: {info.get('description', 'No description')}")
        return 0
    
    logger.info(f"Starting download for model: {args.model}")
    
    try:
        output_path = download_and_convert_model(
            model_name=args.model,
            output_dir=args.output_dir,
            output_filename=args.output_name
        )
        
        logger.info(f"✅ Success! Model saved to: {output_path}")
        logger.info(f"📊 File size: {output_path.stat().st_size / (1024**2):.2f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
