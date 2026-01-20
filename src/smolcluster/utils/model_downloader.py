"""
Utility functions for downloading and managing model weights.
"""
import logging
import urllib.request
import yaml
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model_weights_config() -> Dict:
    """Load model weights configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "model_weights.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model weights config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def download_and_convert_model(
    model_name: str = "gpt2",
    output_dir: Optional[Path] = None,
    output_filename: Optional[str] = None
) -> Path:
    """
    Download a model safetensors file directly from HuggingFace.
    
    Args:
        model_name: Model identifier (e.g., "gpt2", "gpt2-medium")
        output_dir: Directory to save the safetensors file (default: src/data)
        output_filename: Name of the output file (default: from config or <model_name>.safetensors)
    
    Returns:
        Path to the saved safetensors file
    """
    # Load model weights configuration
    config = load_model_weights_config()
    
    # Check if model is in config
    if model_name not in config['models']:
        available = ', '.join(config['models'].keys())
        raise ValueError(f"Model '{model_name}' not found in config. Available: {available}")
    
    model_config = config['models'][model_name]
    download_url = model_config['url']
    
    # Set default paths
    if output_dir is None:
        # Get the project root (4 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "src" / "data"
    
    if output_filename is None:
        output_filename = model_config.get('filename', f"{model_name}.safetensors")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    # Check if file already exists
    if output_path.exists():
        logger.info(f"✅ Model weights already exist at {output_path}")
        return output_path
    
    logger.info(f"📥 Downloading {model_name} ({model_config.get('description', 'Unknown')})...")
    logger.info(f"📍 URL: {download_url}")
    
    try:
        # Download with progress bar
        def download_progress_hook(block_num, block_size, total_size):
            if not hasattr(download_progress_hook, 'pbar'):
                download_progress_hook.pbar = tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {model_name}"
                )
            download_progress_hook.pbar.update(block_size)
        
        # Download the file
        temp_path = output_path.with_suffix('.tmp')
        urllib.request.urlretrieve(download_url, temp_path, download_progress_hook)
        
        # Close progress bar
        if hasattr(download_progress_hook, 'pbar'):
            download_progress_hook.pbar.close()
            delattr(download_progress_hook, 'pbar')
        
        # Move to final location
        temp_path.rename(output_path)
        
        logger.info(f"✅ Successfully downloaded {model_name} to {output_path}")
        logger.info(f"📊 Model size: {output_path.stat().st_size / (1024**2):.2f} MB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        # Clean up temp file if exists
        temp_path = output_path.with_suffix('.tmp')
        if temp_path.exists():
            temp_path.unlink()
        raise


def ensure_model_weights(
    model_identifier: str = "gpt2",
    weights_path: Optional[Path] = None
) -> Path:
    """
    Ensure model weights exist, downloading if necessary.
    
    Args:
        model_identifier: HuggingFace model name
        weights_path: Expected path to weights file
    
    Returns:
        Path to the weights file
    """
    if weights_path is None:
        # Default path
        project_root = Path(__file__).parent.parent.parent.parent
        weights_path = project_root / "src" / "data" / f"{model_identifier}.safetensors"
    
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        logger.warning(f"⚠️  Weights not found at {weights_path}")
        logger.info(f"🔄 Downloading {model_identifier}...")
        
        # Download and convert
        output_dir = weights_path.parent
        output_filename = weights_path.name
        
        return download_and_convert_model(
            model_name=model_identifier,
            output_dir=output_dir,
            output_filename=output_filename
        )
    
    logger.info(f"✅ Found weights at {weights_path}")
    return weights_path


if __name__ == "__main__":
    
    # Test the download functionality
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Download GPT-2
    path = download_and_convert_model("gpt2")
    print(f"Model saved to: {path}")
