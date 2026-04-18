import torch
from torch import nn


class TupleConcatPredictionHead(nn.Module):
    """
    Tuple-level prediction head.

    Expects a tensor for each tuple slot:
        [h_0, h_1, ..., h_{k-1}]
    where each h_i has shape [batch_num_examples, hidden_dim].

    It concatenates them into:
        [batch_num_examples, hidden_dim * k]
    and applies an MLP.

    Parameters
    ----------
    hidden_dim : int
        Embedding size of each tuple member.
    tuple_arity : int
        Number of entities in the tuple.
    out_dim : int
        Output dimension. Use:
          - 1 for binary classification / regression
          - num_classes for multiclass classification
    hidden_dims : tuple[int, ...]
        Hidden layer sizes for the MLP.
    dropout : float
        Dropout probability between MLP layers.
    activation : nn.Module
        Activation class, e.g. nn.ReLU.
    layer_norm : bool
        Whether to apply LayerNorm after each hidden linear layer.
    """
    def __init__(
        self,
        hidden_dim: int,
        tuple_arity: int = 3,
        out_dim: int = 1,
        hidden_dims=(128, 64),
        dropout: float = 0.0,
        activation=nn.ReLU,
        layer_norm: bool = False,
    ):
        super().__init__()

        if tuple_arity <= 0:
            raise ValueError(f"tuple_arity must be positive, got {tuple_arity}")

        self.hidden_dim = hidden_dim
        self.tuple_arity = tuple_arity
        self.out_dim = out_dim

        in_dim = hidden_dim * tuple_arity

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tuple_embeddings):
        """
        Parameters
        ----------
        tuple_embeddings : Sequence[Tensor]
            List/tuple of length `tuple_arity`.
            Each tensor has shape [batch_num_examples, hidden_dim].

        Returns
        -------
        Tensor
            Shape [batch_num_examples, out_dim]
            If out_dim == 1, you may `.squeeze(-1)` outside if desired.
        """
        if len(tuple_embeddings) != self.tuple_arity:
            raise ValueError(
                f"Expected {self.tuple_arity} tuple embeddings, "
                f"got {len(tuple_embeddings)}"
            )

        batch_sizes = [x.size(0) for x in tuple_embeddings]
        if len(set(batch_sizes)) != 1:
            raise ValueError(
                f"All tuple embeddings must share the same batch dimension, "
                f"got {batch_sizes}"
            )

        hidden_dims = [x.size(-1) for x in tuple_embeddings]
        if any(h != self.hidden_dim for h in hidden_dims):
            raise ValueError(
                f"All tuple embeddings must have hidden_dim={self.hidden_dim}, "
                f"got {hidden_dims}"
            )

        z = torch.cat(tuple_embeddings, dim=-1)  # [B, hidden_dim * tuple_arity]
        return self.mlp(z)
    

class TupleConcatPredictor(nn.Module):
    """
    Wrapper around TupleConcatPredictionHead that gathers tuple slot embeddings
    directly from the loader output.

    For homogeneous batches:
        logits = predictor(node_embeddings, batch)

    For heterogeneous batches:
        logits = predictor(paper_embeddings, batch['paper'])
    """
    def __init__(
        self,
        hidden_dim: int,
        tuple_arity: int = 3,
        out_dim: int = 1,
        hidden_dims=(128, 64),
        dropout: float = 0.0,
        activation=nn.ReLU,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.head = TupleConcatPredictionHead(
            hidden_dim=hidden_dim,
            tuple_arity=tuple_arity,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, node_embeddings: torch.Tensor, batch_or_store):
        """
        Parameters
        ----------
        node_embeddings : Tensor
            Local node embeddings of shape [num_local_nodes, hidden_dim]
            from the sampled subgraph.
        batch_or_store :
            Either:
              - homogeneous batch (Data/Batch), or
              - heterogeneous node store, e.g. batch['paper']

            Must expose:
              - input_local_nodes_tuple
              - tuple_arity
        """
        tuple_indices = batch_or_store.input_local_nodes_tuple
        tuple_embeddings = [node_embeddings[idx] for idx in tuple_indices]
        return self.head(tuple_embeddings)