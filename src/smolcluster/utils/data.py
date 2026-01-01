# For spliting the data uniquely across workers
import torch


def get_data_indices(dataset_length: int, world_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(dataset_length, generator=generator)
    split_indices = torch.chunk(indices, world_size)
    return split_indices
