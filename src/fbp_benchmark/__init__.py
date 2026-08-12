"""A facial beauty prediction benchmark: one protocol, twenty methods.

from fbp_benchmark import load_protocol, run

protocol = load_protocol()              # MEBeauty, from the Hub
result = run("cnn-resnet18", protocol)
print(result.metrics)
"""

from .data import DatasetSpec, Protocol, Split, load_protocol
from .registry import available, create, get, register
from .runner import Result, run, save

__all__ = [
    "DatasetSpec",
    "Protocol",
    "Result",
    "Split",
    "available",
    "create",
    "get",
    "load_protocol",
    "register",
    "run",
    "save",
]
