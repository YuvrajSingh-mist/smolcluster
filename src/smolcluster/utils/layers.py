"""Custom neural-network layer helpers and HuggingFace model sharding utilities for distributed inference."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open


from smolcluster.models.moe import ExpertBlock, Router, TextEmbeddings


logger = logging.getLogger(__name__)

model_weights = Path(__file__).parent.parent.parent / "data" / "gpt2.safetensors"


def get_hfmodel_per_node(
    model, num_nodes: int, local_rank: int, model_name: str, total_layers: int
) -> list:
    out_layers = {}
    results = []

    assert local_rank <= num_nodes, "Local rank must be less than number of nodes"

    if model_name == "causal_gpt2":
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
            out_layers["model.transformer.wte"] = model.transformer.wte
            out_layers["model.transformer.wpe"] = model.transformer.wpe

            # Get the indices for this node's layers
            node_layer_indices = split_indices[local_rank].tolist()

            # Add transformer layers for this node
            for layer_idx in node_layer_indices:
                out_layers[f"model.transformer.h.{layer_idx}"] = layers[layer_idx]

        # Add final layers for last node
        elif local_rank == num_nodes - 1:
            # Get the indices for this node's layers
            node_layer_indices = split_indices[local_rank].tolist()

            # Add transformer layers for this node
            for layer_idx in node_layer_indices:
                out_layers[f"model.transformer.h.{layer_idx}"] = layers[layer_idx]

            out_layers["model.transformer.ln_f"] = model.transformer.ln_f
            out_layers["model.lm_head"] = model.lm_head

        else:
            # Get the indices for this node's layers
            node_layer_indices = split_indices[local_rank].tolist()

            # Add transformer layers for this node
            for layer_idx in node_layer_indices:
                out_layers[f"model.transformer.h.{layer_idx}"] = layers[layer_idx]

        # Collect parameter names for loading from safetensors
        for layer_name, layer in out_layers.items():
            for param_name, _param in layer.named_parameters():
                if layer_name == "model.lm_head":
                    results.append(layer_name.split("model.")[1] + "." + param_name)
                else:
                    results.append(
                        layer_name.split("model.transformer.")[1] + "." + param_name
                    )

        # Build mapping for remapping keys to ModuleList indexing

        layer_mapping = {}
        modulelist_idx = 0

        logger.info(f"Loaded layers: {list(out_layers.keys())}")

        for layer_name, _layer in out_layers.items():
            if "h." in layer_name:
                # Extract original layer number (e.g., 'model.transformer.h.9' -> 9)
                original_idx = int(layer_name.split("h.")[1])
                layer_mapping[original_idx] = modulelist_idx
                modulelist_idx += 1
            elif "ln_f" in layer_name:
                layer_mapping["ln_f"] = modulelist_idx
                modulelist_idx += 1
            elif "wte" in layer_name:
                layer_mapping["wte"] = modulelist_idx
                modulelist_idx += 1
            elif "wpe" in layer_name:
                layer_mapping["wpe"] = modulelist_idx
                modulelist_idx += 1
            elif "lm_head" in layer_name:
                layer_mapping["lm_head"] = modulelist_idx
                modulelist_idx += 1

        return layer_mapping, out_layers, results


def load_weights_per_node(
    model_name: str,
    out_layers: dict,
    layer_mapping: dict,
    local_rank: int,
    num_nodes: int,
    results: list[str],
    weights_path: str = model_weights,
) -> torch.nn.ModuleList:
    stage_sd = {}

    # Load weights with remapped keys
    with safe_open(weights_path, framework="pt") as f:
        for k in results:
            for key in f.keys():
                if k == key:
                    # Remap the key to match ModuleList indexing
                    if "h." in k:
                        original_idx = int(k.split("h.")[1].split(".")[0])
                        new_idx = layer_mapping[original_idx]
                        # Replace h.9 with 0, h.10 with 1, etc.
                        new_key = k.replace(f"h.{original_idx}", str(new_idx))
                    elif "ln_f" in k:
                        new_key = k.replace("ln_f", str(layer_mapping["ln_f"]))
                    elif "wpe" in k:
                        new_key = k.replace("wpe", str(layer_mapping["wpe"]))
                    elif "wte" in k:
                        new_key = k.replace("wte", str(layer_mapping["wte"]))
                    elif "lm_head" in k:
                        new_key = k.replace("lm_head", str(layer_mapping["lm_head"]))
                    else:
                        new_key = k

                    print(f"Loaded weights for: {k}")
                    stage_sd[new_key] = f.get_tensor(k)

        if model_name == "causal_gpt2":
            # CRITICAL: For weight tying, load wte weights into lm_head
            # In GPT-2, lm_head.weight should be tied to wte.weight
            # Since they're on different nodes, we load the same weights to both
            if local_rank == num_nodes - 1 and "lm_head" in layer_mapping:
                # Load wte.weight for lm_head.weight (weight tying)
                wte_weight_key = "wte.weight"
                if wte_weight_key in f.keys():
                    lm_head_idx = layer_mapping["lm_head"]
                    stage_sd[f"{lm_head_idx}.weight"] = f.get_tensor(wte_weight_key)
                    print("Weight tying: Loaded wte.weight into lm_head.weight")

        # Now load into ModuleList
        final_layers = list(out_layers.values())
        final_model = torch.nn.ModuleList(final_layers)
        final_model.load_state_dict(stage_sd, strict=False)

        return final_model


def get_model_per_node(
    model,
    num_nodes: int,
    local_rank: int,
    total_layers: int,
    model_type: str = "causal_gpt2",
    num_experts: int = None,
    model_config: Optional[Dict] = None,
) -> Tuple[torch.nn.ModuleList, dict]:
    """Partition model components across nodes.

    For 'causal_gpt2': extracts layer slices from a pre-built model.
    For 'causal_mixtral' (Expert Parallelism): builds rank-specific components
        from scratch — no full model required; pass model=None.

    Args:
        model:        Full model (causal_gpt2) or None (causal_mixtral).
        num_nodes:    Total number of worker nodes.
        local_rank:   This node's rank.
        total_layers: Number of decoder layers.
        model_type:   'causal_gpt2' or 'causal_mixtral'.
        num_experts:  Number of MoE experts (causal_mixtral only).
        model_config: Dict of hyperparams for causal_mixtral:
                      vocab_size, embedding_dims, top_k, no_of_heads, device,
                      attn_dropout, dropout, noisy_topk.

    Returns:
        (ModuleList of all local modules, out_layers dict keyed by component name)
    """
    out_layers = {}
    assert local_rank < num_nodes, "local_rank must be less than num_nodes"

    if model_type == "causal_gpt2":
        layers = list(model.blocks)

        layer_indices = torch.arange(total_layers)
        split_indices = torch.chunk(layer_indices, num_nodes)
        logger.info(f"Layer splits: {split_indices}")

        assert len(split_indices) <= num_nodes

        # Add embeddings for first node
        if local_rank == 0:
            out_layers["model.token_embedding"] = model.token_embedding
            out_layers["model.position_embedding"] = model.position_embedding

            node_layer_indices = split_indices[local_rank].tolist()
            for layer_idx in node_layer_indices:
                out_layers[f"model.blocks.{layer_idx}"] = layers[layer_idx]

        # Add final layers for last node
        elif local_rank == num_nodes - 1:
            node_layer_indices = split_indices[local_rank].tolist()
            for layer_idx in node_layer_indices:
                out_layers[f"model.blocks.{layer_idx}"] = layers[layer_idx]

            out_layers["model.ln_f"] = model.ln_f
            out_layers["model.lm_head"] = model.lm_head

        else:
            node_layer_indices = split_indices[local_rank].tolist()
            for layer_idx in node_layer_indices:
                out_layers[f"model.blocks.{layer_idx}"] = layers[layer_idx]

    elif model_type == "causal_mixtral":
        # Expert Parallelism: extract components from the passed Mixtral model.
        # Last rank gets the full Mixtral transformer; all ranks get ExpertBlocks
        # built from model attributes; rank 0 additionally gets TextEmbeddings + Router.
        
        assert num_experts is not None, "num_experts required for causal_mixtral"
        assert model_config is not None, "model_config required for causal_mixtral"

        # Read arch params from the model itself (consistent with GPT2 pattern)
        vocab_size     = model.vocab_size
        embedding_dims = model.embeddings_dims
        top_k          = model_config["top_k"]
        device         = model_config["device"]
        noisy_topk     = model_config.get("noisy_topk", True)

        expert_shard_indices = get_expert_per_node(local_rank, num_nodes, num_experts)
      
        # Rank 0: TextEmbeddings + Router (built fresh — not part of Mixtral)
        if local_rank == 0:
            out_layers["model.text_embeddings"] = TextEmbeddings(
                vocab_size=vocab_size, embeddings_dims=embedding_dims, device=device
            )
            out_layers["model.router"] = Router(
                embeddings_dims=embedding_dims, num_experts=num_experts,
                top_k=top_k, device=device, noisy_topk=noisy_topk,
            )
            logger.info(f"Rank 0: built TextEmbeddings + Router")

        # All ranks: ExpertBlocks for their shard (built fresh — not part of Mixtral)
        for idx in expert_shard_indices:
            out_layers[f"model.expert_{idx}"] = ExpertBlock(embedding_dims, device)
        logger.info(f"Rank {local_rank}: built {len(expert_shard_indices)} ExpertBlocks (dim={embedding_dims})")

        # Last rank: the full Mixtral transformer (attention + layernorm + lm_head)
        if local_rank == num_nodes - 1:
            out_layers["model.mixtral"] = model
            logger.info(f"Rank {local_rank}: assigned Mixtral transformer from passed model")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: 'causal_gpt2', 'causal_mixtral'")

    logger.info(f"Rank {local_rank} layers: {list(out_layers.keys())}")
    return torch.nn.ModuleList(list(out_layers.values())), out_layers


def get_expert_per_node(local_rank: int, num_nodes: int, num_experts: int) -> List[int]:
    assert local_rank < num_nodes, "local rank must be less than number of total nodes"
  

    expert_indices = torch.arange(num_experts)
    split_indices = torch.chunk(expert_indices, num_nodes)
    logger.info(f"Expert splits: {split_indices}")

    return split_indices[local_rank].tolist()



