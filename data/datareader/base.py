from abc import abstractmethod, ABC

from relbench.base.database import Database


class BaseDataReader(ABC):
    """
    Base class for OCEL data readers that produce an EventGraph (PyG HeteroData).
    """

    def __init__(self, path: str, *args, **kwargs) -> None:
        self.path = path

    @abstractmethod
    def parse_tables(
        self,
        dataset_name: str | None,
    ) -> Database:
        """Parse the data file and return an EventGraph."""
        pass
