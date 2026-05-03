import random
from typing import TypeVar

T = TypeVar("T")


def flatten(lst: list[list[T]]) -> list[T]:
    """Flattens a list of lists into a single list."""
    return [item for sublist in lst for item in sublist]


def batch(lst: list[T], size: int) -> list[list[T]]:
    """Groups list into batches of specified size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def mix_lists(
    list1: list[T],
    list2: list[T],
    ratio1: float,
    total_samples: int,
    seed: int = 42,
) -> list[T]:
    """Mix two lists at specified ratio.
    
    Args:
        list1: First list to sample from
        list2: Second list to sample from
        ratio1: Fraction of data that should come from list1 (0.0 to 1.0)
        total_samples: Total number of samples in the mixed list
        seed: Random seed for reproducibility
        
    Returns:
        Mixed list with specified ratio and total samples
        
    Raises:
        ValueError: If either list is too small to provide the required samples
    """
    rng = random.Random(seed)
    
    # Calculate number of samples for each list
    n_from_list1 = int(total_samples * ratio1)
    n_from_list2 = total_samples - n_from_list1
    
    # Check if lists are large enough
    if len(list1) < n_from_list1:
        raise ValueError(
            f"List1 too small: need {n_from_list1} samples, "
            f"but only have {len(list1)} samples"
        )
    
    if len(list2) < n_from_list2:
        raise ValueError(
            f"List2 too small: need {n_from_list2} samples, "
            f"but only have {len(list2)} samples"
        )
    
    # Sample from each list
    sampled_from_list1 = rng.sample(list1, n_from_list1)
    sampled_from_list2 = rng.sample(list2, n_from_list2)
    
    # Combine and shuffle
    mixed_list = sampled_from_list1 + sampled_from_list2
    rng.shuffle(mixed_list)
    
    return mixed_list