import os


def get_all_filepaths(directory):
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepaths.append(os.path.join(root, file))
    return filepaths


def get_first_level_subdirectories(directory):
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdirectories.append(entry.path)
    return subdirectories


def get_all_subdirectories(directory):
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subdirectories.append(os.path.join(root, dir))
    return subdirectories
