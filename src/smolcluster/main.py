import torch
from typing import Dict

def all_reduce(grads_dict: Dict[int, Dict[str, torch.Tensor]], num_workers_connected: int) -> Dict[str, torch.Tensor]:
    # Copy of the function for testing (fix as needed)
    for name in grads_dict:
        summed_grad = torch.zeros_like(list(grads_dict[name].values())[0])
        for _, worker_grads in grads_dict[name].items():
            summed_grad += worker_grads[name]
        grads_dict[name] = summed_grad / num_workers_connected
    
    return grads_dict

if __name__ == "__main__":
    # Mock data: 2 workers, 2 params ('weight', 'bias')
    grads_dict = {
        0: {'weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 'bias': torch.tensor([0.1, 0.2])},
        1: {'weight': torch.tensor([[0.5, 1.5], [2.5, 3.5]]), 'bias': torch.tensor([0.05, 0.15])}
    }
    num_workers = 2
    
    print("Input grads_dict:")
    for worker, grads in grads_dict.items():
        print(f"Worker {worker}: {grads}")
    
    result = all_reduce(grads_dict, num_workers)
    print("\nOutput averaged grads:")
    print(result)
    
    # Expected: Averaged tensors, e.g., weight: [[0.75, 1.75], [2.75, 3.75]], bias: [0.075, 0.175]