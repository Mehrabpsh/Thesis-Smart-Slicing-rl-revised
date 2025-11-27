"""Serialization utilities."""

import json
import pickle
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path) -> None:
    """Save object to JSON file.
    
    Args:
        obj: Object to save (must be JSON serializable)
        path: Path to save file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    """Load object from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded object
    """
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to pickle
        path: Path to save file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load object from pickle file.
    
    Args:
        path: Path to pickle file
        
    Returns:
        Unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)

