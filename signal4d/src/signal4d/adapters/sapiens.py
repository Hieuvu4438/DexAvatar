from .file_cache import NpzEstimatorAdapter


class SapiensAdapter(NpzEstimatorAdapter):
    def __init__(self, root: str) -> None:
        super().__init__("sapiens", root)
