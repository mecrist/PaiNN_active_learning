#!/usr/bin/env python
"""
Dataset I/O and Management Utilities.
Handles reading, writing, deduplicating, and splitting training and validation xyz sets.
"""

from pathlib import Path
from typing import List, Union, Set
from ase import Atoms
from ase.io import read, write


def load_dataset(filepath: Union[str, Path]) -> List[Atoms]:
    """Loads all frames from an extxyz dataset file."""
    path = Path(filepath)
    if not path.exists():
        return []
    return read(str(path), index=":", format="extxyz")


def save_labeled_point(filepath: Union[str, Path], atoms: Atoms, append: bool = True):
    """Appends or writes a labeled Atoms frame to an extxyz file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="extxyz", append=append)


def get_energy_fingerprints(frames: List[Atoms], decimals: int = 4) -> Set[float]:
    """Extracts rounded reference energies for fast deduplication check."""
    fps = set()
    for at in frames:
        e = at.info.get("REF_energy", at.info.get("energy", None))
        if e is not None:
            fps.add(round(float(e), decimals))
    return fps
