from transformers import AutoConfig
from typing import List, Dict
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
    
    

        
def set_layer_weights(model, layers: Dict[str, torch.Tensor]) -> dict:
    
    results = []
    
    for layer in layers:
        for name, param in layers[layer].named_parameters():
            results.append('h.' + layer_name.split('.')[-1] + name)
            
    
    stage_sd = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            if k in results:
                stage_sd[k] = f.get_tensor(k)
    delete_layer = []
    for name, params in model.named_parameters():
        
        curr_name = 'h.' + name.split('.')[-1]
        if curr_name in list(stage_sd.keys()):
            params.data = stage_sd[name].cpu().clone()
        else:
            delete_layer.append(name)
    
    for layer in delete_layer:
        for name, param in model.named_parameters():
             if layer == name:
                 del name
    return model
        
        
        
        
        