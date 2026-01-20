import logging
from transformers import AutoConfig
from typing import List
import torch
from safetensors import safe_open
from pathlib import Path

logger = logging.getLogger(__name__)

model_weights = Path(__file__).parent.parent.parent / "data" / "gpt2.safetensors"
def get_model_per_node(model, num_nodes: int, local_rank: int, model_name: str, total_layers: int) -> List:
    
    out_layers = {}
    results = []
    
    assert local_rank < num_nodes, "Local rank must be less than number of nodes"
         
    if model_name == 'causal_gpt2':
        
        # Collect all transformer layers
        layers = list(model.transformer.h)
        
        # Create indices for all layers and split them across nodes using torch.chunk
        # This handles uneven splits automatically
        layer_indices = torch.arange(total_layers)
        split_indices = torch.chunk(layer_indices, num_nodes)
        logger.info(f"Layer splits: {split_indices}")
       
        assert len(split_indices) <= num_nodes
        
        # Add embeddings for first node
        if local_rank == 0:
            out_layers['model.transformer.wte'] = model.transformer.wte
            out_layers['model.transformer.wpe'] = model.transformer.wpe

            # Get the indices for this node's layers
            node_layer_indices = split_indices[local_rank].tolist()
            
            # Add transformer layers for this node
            for layer_idx in node_layer_indices:
                out_layers[f'model.transformer.h.{layer_idx}'] = layers[layer_idx]
        
        # Add final layers for last node
        elif local_rank == num_nodes - 1:
            # Get the indices for this node's layers
            node_layer_indices = split_indices[local_rank].tolist()
            
            # Add transformer layers for this node
            for layer_idx in node_layer_indices:
                out_layers[f'model.transformer.h.{layer_idx}'] = layers[layer_idx]
            
            out_layers['model.transformer.ln_f'] = model.transformer.ln_f   
            out_layers['model.lm_head'] = model.lm_head
        
        else:
            # Get the indices for this node's layers
            node_layer_indices = split_indices[local_rank].tolist()
            
            # Add transformer layers for this node
            for layer_idx in node_layer_indices:
                out_layers[f'model.transformer.h.{layer_idx}'] = layers[layer_idx]
                
        # Collect parameter names for loading from safetensors
        for layer_name, layer in out_layers.items():
            for param_name, param in layer.named_parameters():
                if layer_name == 'model.lm_head':
                    results.append(layer_name.split('model.')[1] + '.' + param_name)
                else:
                    results.append(layer_name.split('model.transformer.')[1] + '.' + param_name)
        
        # Build mapping for remapping keys to ModuleList indexing
        
        layer_mapping = {}
        modulelist_idx = 0
        
        logger.info(f"Loaded layers: {list(out_layers.keys())}")
        
        for layer_name, layer in out_layers.items():
            if 'h.' in layer_name:
                # Extract original layer number (e.g., 'model.transformer.h.9' -> 9)
                original_idx = int(layer_name.split('h.')[1])
                layer_mapping[original_idx] = modulelist_idx
                modulelist_idx += 1
            elif 'ln_f' in layer_name:
                layer_mapping['ln_f'] = modulelist_idx
                modulelist_idx += 1
            elif 'wte' in layer_name:
                layer_mapping['wte'] = modulelist_idx
                modulelist_idx += 1
            elif 'wpe' in layer_name:
                layer_mapping['wpe'] = modulelist_idx
                modulelist_idx += 1
            elif 'lm_head' in layer_name:
                layer_mapping['lm_head'] = modulelist_idx
                modulelist_idx += 1
                
        return layer_mapping, out_layers, results
    
    
def load_weights_per_node(model_name: str = model_weights, weights_path: str, out_layers: dict, layer_mapping: dict, local_rank: int, num_nodes: int, results: List[str]) -> torch.nn.ModuleList: 
    
    stage_sd = {}
    
    # Load weights with remapped keys
    with safe_open(weights_path, framework="pt") as f:
        for k in results:
            for key in f.keys():
                if k == key:
                    # Remap the key to match ModuleList indexing
                    if 'h.' in k:
                        original_idx = int(k.split('h.')[1].split('.')[0])
                        new_idx = layer_mapping[original_idx]
                        # Replace h.9 with 0, h.10 with 1, etc.
                        new_key = k.replace(f'h.{original_idx}', str(new_idx))
                    elif 'ln_f' in k:
                        new_key = k.replace('ln_f', str(layer_mapping['ln_f']))
                    elif 'wpe' in k:
                        new_key = k.replace('wpe', str(layer_mapping['wpe']))
                    elif 'wte' in k:
                        new_key = k.replace('wte', str(layer_mapping['wte']))
                    elif 'lm_head' in k:
                        new_key = k.replace('lm_head', str(layer_mapping['lm_head']))
                    else:
                        new_key = k
                        
                    print(f"Loaded weights for: {k}")
                    stage_sd[new_key] = f.get_tensor(k)
        
        if model_name == 'causal_gpt2':
            # CRITICAL: For weight tying, load wte weights into lm_head
            # In GPT-2, lm_head.weight should be tied to wte.weight
            # Since they're on different nodes, we load the same weights to both
            if local_rank == num_nodes - 1 and 'lm_head' in layer_mapping:
                # Load wte.weight for lm_head.weight (weight tying)
                wte_weight_key = 'wte.weight'
                if wte_weight_key in f.keys():
                    lm_head_idx = layer_mapping['lm_head']
                    stage_sd[f'{lm_head_idx}.weight'] = f.get_tensor(wte_weight_key)
                    print(f"Weight tying: Loaded wte.weight into lm_head.weight")

        # Now load into ModuleList
        final_layers = list(out_layers.values())
        final_model = torch.nn.ModuleList(final_layers)
        loaded_model = final_model.load_state_dict(stage_sd, strict=False)

        return final_model