from transformers import AutoConfig
from typing import List
import torch
from safetensors import safe_open
from pathlib import Path


safetensors_path = Path(__file__).parent.parent.parent / "data" / "model.safetensors"

def get_layers_per_node(config: AutoConfig, model, num_nodes: int, local_rank: int, model_name: str) -> List:
    
    assert num_nodes > local_rank, "Number of nodes must be greater than local rank"
    
    out_layers = {}
    
    if model_name == 'causal_gpt2':
        
        total_layers = config.n_layer
        
        layers_per_node = total_layers // num_nodes
        
        layers = []
        
        for layer in model.transformer.h:
            layers.append(layer)     
            
        if local_rank == 0:
            out_layers['model.transformer.wte'] = model.transformer.wte
            out_layers['model.transformer.wpe'] = model.transformer.wpe
   
        
        elif local_rank == num_nodes - 1:
            selected_layers = layers[layers_per_node * local_rank : ((layers_per_node * local_rank) + layers_per_node) - 1]
            
            for layer in selected_layers:
                out_layers[f'model.transformer.h.{layers.index(layer)}'] = layer
                
            out_layers['model.transformer.ln_f'] = model.transformer.ln_f   
        
        
        else:
            selected_layers = layers[layers_per_node * local_rank : ((layers_per_node * local_rank) + layers_per_node) - 1]
            
            for layer in selected_layers:
                out_layers[f'model.transformer.h.{layers.index(layer)}'] = layer
        
        return out_layers
    
    

        
def get_layer_weights(layers: dict[str, torch.Tensor], layer_name: str) -> dict:
    
    results = []
    
    for i in layers['model.transformer.h.10'].named_parameters():
        results.append('h.' + layer_name.split('.')[-1] + i[0])
        
    
    stage_sd = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            if k in results:
                stage_sd[k] = f.get_tensor(k)
        
    return stage_sd
        
        
        
        
        