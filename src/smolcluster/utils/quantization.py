"""
Quantization utilities for model weight compression in distributed training.
Uses W8A16 (8-bit weights, 16-bit activations) quantization for efficient network transfer.
"""

from typing import Union

import torch


def linear_quantize(tensor, dtype=torch.int8):
    """
    Quantize a tensor to int8 using linear quantization.

    Args:
        tensor: Input tensor to quantize
        dtype: Target dtype (default: torch.int8)

    Returns:
        tuple: (scale, zero_point, quantized_tensor)
    """
    r_min = tensor.min().item()
    r_max = tensor.max().item()

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    # Handle edge case: tensor has very small range
    if r_max - r_min <= 1e-5:
        scale = 2 * r_max / (q_max - q_min) if r_max != 0 else 1e-5
        zero_point = torch.tensor(0, device=tensor.device, dtype=torch.int8)
    else:
        scale = (r_max - r_min) / (q_max - q_min)
        zero_point = torch.round(
            torch.tensor(q_min - (r_min / scale), device=tensor.device)
        )
        zero_point = torch.clamp(zero_point, min=q_min, max=q_max).to(torch.int8)

    # Quantize
    quantized_tensor = torch.round(tensor / scale + zero_point.float())
    quantized_tensor = torch.clamp(quantized_tensor, min=q_min, max=q_max).to(dtype)

    return scale, zero_point.item(), quantized_tensor


def channel_linear_quantize(tensor, dim=0, dtype=torch.int8):
    """
    Per-channel quantization for better precision.

    Args:
        tensor: Input tensor (2D)
        dim: Channel dimension (0 for rows, 1 for columns)
        dtype: Target dtype (default: torch.int8)

    Returns:
        tuple: (scales, zero_points, quantized_tensor)
    """
    device = tensor.device
    num_channels = tensor.size(dim)

    scales = torch.zeros(num_channels, device=device)
    zero_pts = torch.zeros(num_channels, dtype=torch.int8, device=device)
    quantized_tensor = torch.zeros_like(tensor, dtype=dtype)

    # Quantize each channel independently
    for i in range(num_channels):
        channel = tensor.select(dim, i)
        scales[i], zero_pts[i], quant = linear_quantize(channel, dtype=dtype)

        if dim == 1:
            quantized_tensor[:, i] = quant
        else:
            quantized_tensor[i, :] = quant

    # Reshape scales and zero points for broadcasting
    if dim == 0:
        scales = scales.view(num_channels, 1)
        zero_pts = zero_pts.view(num_channels, 1)
    elif dim == 1:
        scales = scales.view(1, num_channels)
        zero_pts = zero_pts.view(1, num_channels)

    return scales, zero_pts, quantized_tensor


def linear_dequantize(scale, zero_point, quantized_tensor):
    """
    Dequantize an int8 tensor back to float32.

    Args:
        scale: Quantization scale (float or tensor)
        zero_point: Zero point (int or tensor)
        quantized_tensor: Quantized int8 tensor

    Returns:
        torch.Tensor: Dequantized float32 tensor
    """
    if isinstance(zero_point, int):
        zero_point = torch.tensor(zero_point, device=quantized_tensor.device)

    return scale * (quantized_tensor.float() - zero_point.float())


def quantize_model_weights(weights_dict):
    """
    Quantize all weights in a model state dict.

    Args:
        weights_dict: Dictionary of {layer_name: weight_tensor}

    Returns:
        dict: Quantized weights with metadata
            {
                'layer_name': {
                    'quantized': int8 tensor,
                    'scales': scale tensor,
                    'zero_points': zero point tensor,
                    'shape': original shape,
                    'dim': quantization dimension (0 for 2D, None for 1D)
                }
            }
    """
    quantized_dict = {}

    for name, weight in weights_dict.items():
        # Move to CPU for serialization
        weight_cpu = weight.cpu()

        if weight_cpu.dim() == 2:
            # 2D tensor (e.g., Linear layer): use per-channel quantization
            scales, zero_pts, quant = channel_linear_quantize(weight_cpu, dim=0)
            quantized_dict[name] = {
                "quantized": quant,
                "scales": scales,
                "zero_points": zero_pts,
                "shape": weight.shape,
                "dim": 0,
            }
        elif weight_cpu.dim() == 1:
            # 1D tensor (e.g., bias): use simple quantization
            scale, zero_pt, quant = linear_quantize(weight_cpu)
            quantized_dict[name] = {
                "quantized": quant,
                "scales": torch.tensor([scale]),
                "zero_points": torch.tensor([zero_pt], dtype=torch.int8),
                "shape": weight.shape,
                "dim": None,
            }
        else:
            # Higher dimensional tensors (e.g., Conv): flatten first
            original_shape = weight_cpu.shape
            flattened = weight_cpu.view(weight_cpu.size(0), -1)
            scales, zero_pts, quant = channel_linear_quantize(flattened, dim=0)
            quantized_dict[name] = {
                "quantized": quant,
                "scales": scales,
                "zero_points": zero_pts,
                "shape": original_shape,
                "dim": 0,
            }

    return quantized_dict


def dequantize_model_weights(
    quantized_dict: dict[str, torch.Tensor], device: Union[str, torch.device] = "cpu"
):
    """
    Dequantize a full model state dict.

    Args:
        quantized_dict: Dictionary from quantize_model_weights()
        device: Target device for dequantized weights (str or torch.device)

    Returns:
        dict: {layer_name: dequantized_weight_tensor}
    """
    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)

    weights_dict = {}

    for name, data in quantized_dict.items():
        quant = data["quantized"].to(device)
        scales = data["scales"].to(device)
        zero_pts = data["zero_points"].to(device)
        original_shape = data["shape"]

        # Dequantize
        dequantized = linear_dequantize(scales, zero_pts, quant)

        # Reshape to original shape if needed
        if dequantized.shape != original_shape:
            dequantized = dequantized.view(original_shape)

        weights_dict[name] = dequantized

    return weights_dict


def calculate_compression_ratio(original_dict, quantized_dict):
    """
    Calculate compression ratio achieved by quantization.

    Args:
        original_dict: Original weights dict {name: tensor}
        quantized_dict: Quantized weights dict from quantize_model_weights()

    Returns:
        dict: {'ratio': float, 'original_mb': float, 'compressed_mb': float}
    """
    # Calculate original size (float32 = 4 bytes)
    original_size = sum(w.numel() * 4 for w in original_dict.values())

    # Calculate quantized size (int8 = 1 byte + scales/zero_points overhead)
    quantized_size = 0
    for data in quantized_dict.values():
        quantized_size += data["quantized"].numel() * 1  # int8
        quantized_size += data["scales"].numel() * 4  # float32 scales
        quantized_size += data["zero_points"].numel() * 1  # int8 zero points

    ratio = original_size / quantized_size if quantized_size > 0 else 0

    return {
        "ratio": ratio,
        "original_mb": original_size / (1024 * 1024),
        "compressed_mb": quantized_size / (1024 * 1024),
    }
