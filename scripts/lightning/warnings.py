import logging
import warnings


def configure_training_warnings() -> None:
    """Silence known noisy third-party warnings in Lightning/HPO runs."""
    for message in [
        "Weights only load failed.*",
        "The given NumPy array is not writable.*",
        "The '.*_dataloader' does not have many workers.*",
        "Checkpoint directory .* exists and is not empty.*",
        "GPU available but not used.*",
    ]:
        warnings.filterwarnings("ignore", message=message, category=UserWarning)

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
