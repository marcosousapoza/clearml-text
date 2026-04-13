"""Lightning entrypoint package for entity-task GNN training."""

import scripts; scripts.load_env()

from .cli import main

__all__ = ["main"]
