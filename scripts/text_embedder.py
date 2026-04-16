from typing import List, Optional

import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class SentenceTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device.type if device else 'cpu',
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)
