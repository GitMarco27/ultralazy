from typing import List, Tuple


def filter_estimators_by_keys(
    keys: list | str, estimators: List[Tuple[str, object]]
) -> List[Tuple[str, object]]:
    """Filter estimators by keys keeping only the filtered ones

    Args:
        keys (list | str): name of estimators to filter
    """
    if isinstance(keys, str):
        keys = list(keys)

    filtered_estimators = []

    for name, model in estimators:
        if name in keys:
            filtered_estimators.append((name, model))
            keys.remove(name)

    if len(keys) > 0:
        print(f"The following estimators were not found: {keys}")

    return filtered_estimators
