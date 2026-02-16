import json
import os
import os.path as op
import pickle  # nosec B403


def create_dir_if_not_exists(dir_name: str) -> None:
    """
    Creates a directory and all necessary parent directories if it does not already exist.

    Args:
        dir_name (str): The path of the directory to create.
    """
    if not op.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def add_suffix_to_path(out_path: str, suffix: str) -> str:
    """
    Injects a suffix into a file path right before its extension.

    Args:
        out_path (str): The original file path.
        suffix (str): The string to append to the filename.

    Returns:
        str: The newly constructed file path.
    """
    base_path = op.basename(out_path)
    dir_path = op.dirname(out_path)
    name_parts, ext = base_path.split(".")[:-1], base_path.split(".")[-1]
    name = ".".join(name_parts)
    return op.join(dir_path, "%s_%s.%s" % (name, suffix, ext))


def write_json(h: dict, file_name: str) -> None:
    """
    Serializes a Python dictionary to a JSON file.

    Args:
        h (dict): The dictionary object to serialize.
        file_name (str): The path to the output JSON file.
    """
    with open(file_name, "w") as f:
        json.dump(h, f, indent=4)


def write_pickle(obj: dict, file_name: str, protocol: int = 3) -> None:
    """
    Serializes a Python object to a binary file using pickle.

    Args:
        obj (dict): The object to serialize.
        file_name (str): The path to the output pickle file.
        protocol (int, optional): The pickle protocol version to use. Defaults to 3.
    """
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(file_name: str) -> dict:
    """
    Deserializes a Python object from a binary pickle file.

    Args:
        file_name (str): The path to the pickle file to read.

    Returns:
        Any: The deserialized Python object.
    """
    with open(file_name, "rb") as f:
        return pickle.load(f)  # nosec B301
