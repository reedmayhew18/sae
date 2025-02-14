import os
from typing import List

def get_file_names(filter_prefix: str, root_dir: str) -> List[str]:
    """
    Walk through root_dir and return a sorted list of file paths whose names start with filter_prefix.
    """
    file_names = []
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith(filter_prefix):
                file_names.append(os.path.join(dirname, filename))
    return sorted(file_names)
