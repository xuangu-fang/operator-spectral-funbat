"""Geometry-aware functional tensor research POC."""

from .bases import BasisSpec, evaluate_basis
from .data import FieldDataset, load_dataset
from .masks import ObservationSplit, make_observation_split
from .models import build_model
from .functional_tucker import (
    DomainKernelFunctionalTucker,
    GeometryConditionedNeuralFunctionalCP,
    GeometryConditionedNeuralFunctionalTucker,
)

__all__ = [
    "BasisSpec",
    "FieldDataset",
    "ObservationSplit",
    "build_model",
    "DomainKernelFunctionalTucker",
    "evaluate_basis",
    "GeometryConditionedNeuralFunctionalCP",
    "GeometryConditionedNeuralFunctionalTucker",
    "load_dataset",
    "make_observation_split",
]
