import os


def get_all_filepaths(directory, extensions=None):
    """
    Get all file paths in a directory, optionally filtered by file extensions.

    Args:
        directory (str): The directory to search for files.
        extensions (list of str, optional): List of file extensions to filter by. Defaults to None.

    Returns:
        list of str: List of file paths.
    """
    filepaths = []
    for root, dirs, files in os.walk(directory):
        filepaths.extend(
            os.path.join(root, file)
            for file in files
            if extensions is None or file.endswith(tuple(extensions))
        )
    return filepaths


def get_first_level_subdirectories(directory):
    """
    Get the first-level subdirectories in a directory.

    Args:
        directory (str): The directory to search for subdirectories.

    Returns:
        list of str: List of first-level subdirectory paths.
    """
    subdirectories = []
    subdirectories.extend(
        entry.path for entry in os.scandir(directory) if entry.is_dir()
    )
    return subdirectories


def get_all_subdirectories(directory):
    """
    Get all subdirectories in a directory.

    Args:
        directory (str): The directory to search for subdirectories.

    Returns:
        list of str: List of all subdirectory paths.
    """
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        subdirectories.extend(os.path.join(root, dir) for dir in dirs)
    return subdirectories


def remove_string_from_filenames_in_directory(directory, removed_string):
    """
    Remove a specific string from filenames in a directory.

    Args:
        directory (str): The directory containing the files.
        removed_string (str): The string to remove from filenames.

    Returns:
        None
    """
    for filename in os.listdir(directory):
        if removed_string in filename:
            new_filename = filename.replace(removed_string, "")
            os.rename(
                os.path.join(directory, filename), os.path.join(directory, new_filename)
            )
