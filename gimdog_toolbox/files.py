import os
from typing import List, Optional

def get_all_filepaths(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    filepaths = []
    for root, dirs, files in os.walk(directory):
        filepaths.extend(
            os.path.join(root, file)
            for file in files
            if extensions is None or file.endswith(tuple(extensions))
        )
    return filepaths

def get_first_level_subdirectories(directory: str) -> List[str]:
    subdirectories = []
    subdirectories.extend(entry.path for entry in os.scandir(directory) if entry.is_dir())
    return subdirectories

def get_all_subdirectories(directory: str) -> List[str]:
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        subdirectories.extend(os.path.join(root, dir) for dir in dirs)
    return subdirectories

def remove_string_from_filenames_in_directory(directory: str, removed_string: str) -> None:
    for filename in os.listdir(directory):
        if removed_string in filename:
            new_filename = filename.replace(removed_string, "")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))