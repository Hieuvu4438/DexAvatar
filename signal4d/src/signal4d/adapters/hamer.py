from .file_cache import NpzEstimatorAdapter


class HaMeRAdapter(NpzEstimatorAdapter):
    def __init__(self, root: str) -> None:
        super().__init__("hamer", root)
