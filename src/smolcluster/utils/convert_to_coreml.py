import torch
from transformers import AutoModelForCausalLM
from pathlib import Path
from typing import Optional
import torch.nn as nn
import coremltools as ct
import os 
import yaml
from huggingface_hub import HfApi, create_repo
from smolcluster.utils.layers import get_hfmodel_per_node
from dotenv import load_dotenv
import numpy as np


# Load environment variables from .env
load_dotenv()

coremlmodel_path = Path(__file__).parent.parent.parent / "data" / "coremlmodel"


def upload_to_huggingface(
    model_path: str,
    repo_id: str = "YuvrajSingh9886/tablet_model_parallelism",
    commit_message: Optional[str] = None,
    private: bool = False
) -> bool:
    """Upload CoreML model to Hugging Face Hub.
    
    Args:
        model_path: Path to the .mlpackage directory
        repo_id: Hugging Face repository ID (format: username/repo-name)
        commit_message: Custom commit message
        private: Whether to create a private repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        model_name = Path(model_path).name  # pyright: ignore[reportAssignmentType]
        
        if commit_message is None:
            commit_message = f"Upload {model_name}"
        
        # Get token from environment variable
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN not found in .env file")
        
        print(f"Initializing Hugging Face API...")
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            print(f"Creating repository {repo_id} (if it doesn't exist)...")
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
                token=hf_token
            )
            print(f"✅ Repository {repo_id} is ready")
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Upload the model folder
        print(f"Uploading {model_name} to Hugging Face Hub...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"models/{model_name}",
            commit_message=commit_message,
        )
        
        print(f"✅ Successfully uploaded {model_name} to https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        return False

class GPT2Shard(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)[0]
            x = x[0] if isinstance(x, tuple) else x
        return x
    
def convert_to_coreml_weights(
    local_rank: int = 0,
    hf_model_identifier: str = 'openai-community/gpt2',
    cluster_config_path: Optional[str] = None,
    nn_config: Optional[dict] = None,
    upload_to_hf: bool = False) -> None:
    
    # Load cluster config
    if cluster_config_path is None:
        config_dir = Path(__file__).parent.parent / "configs"
        cluster_config_path = str(config_dir / "cluster_config_mp.yaml")
    
    with open(cluster_config_path) as f:
        cluster_config = yaml.safe_load(f)
    
    if nn_config is None:
        config_dir = Path(__file__).parent.parent / "configs"
        with open(config_dir / "gpt_config.yaml") as f:
            nn_config = yaml.safe_load(f)
            
    # Extract parameters from config
    num_nodes = cluster_config['num_nodes']
    num_layers = cluster_config['num_layers']
    model_name = cluster_config['model_name']
    max_seq = nn_config["max_seq_len"]
    
    model = AutoModelForCausalLM.from_pretrained(hf_model_identifier)    
    model.eval()
    
    layer_mapping, out_layers, results = get_hfmodel_per_node(
        model,
        num_nodes=num_nodes,
        local_rank=local_rank,
        model_name=model_name,
        total_layers=num_layers
    )
    
    extracted_layers = []
    
    for layer_name, layer in out_layers.items():
        
        for model_layer in model.named_parameters():
            if model_layer[0] == layer_name:
                extracted_layers.append(layer)
                print(f"Set layer {layer_name} weights")
                break
    
    sharded_model = GPT2Shard(extracted_layers)
    input_data = torch.rand(1, 1, 768)  # Example input tensor, adjust shape as needed
    
    traced = torch.jit.trace(sharded_model, input_data)
    
    mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(
            name="x",
            shape=(1, ct.RangeDim(1, max_seq), 768),
            dtype=np.float32
        )
    ],
     
    compute_units=ct.ComputeUnit.CPU_AND_GPU,  # <-- KEY
    minimum_deployment_target=ct.target.iOS17,
)

    
    # Ensure we have an MLModel (not Program)
    if not isinstance(mlmodel, ct.models.MLModel):
        raise TypeError(f"Expected MLModel, got {type(mlmodel)}")
    
        
    os.makedirs(coremlmodel_path, exist_ok=True)
    save_path = str(coremlmodel_path / f"{hf_model_identifier.split('/')[-1]}_rank{local_rank}.mlpackage")
    
    mlmodel.save(save_path)
    print(f"CoreML model saved to: {save_path}")
    
    # Upload to Hugging Face if requested
    if upload_to_hf:
        print(f"\n📤 Uploading to Hugging Face...")
        upload_to_huggingface(
            model_path=save_path,
            commit_message=f"Add {hf_model_identifier.split('/')[-1]} rank {local_rank} CoreML model"
        )
    



if __name__ == '__main__':
    
    # Convert all ranks for the cluster
    with open(Path(__file__).parent.parent / "configs" / "cluster_config_mp.yaml") as f:
        cluster_config = yaml.safe_load(f)
    
    num_nodes = cluster_config['num_nodes']
    
    for rank in range(num_nodes):
        print(f"\n{'='*60}")
        print(f"Converting rank {rank}/{num_nodes-1}")
        print(f"{'='*60}\n")
        
        convert_to_coreml_weights(
            local_rank=rank,
            upload_to_hf=True  # Set to True to upload to Hugging Face
        )
    