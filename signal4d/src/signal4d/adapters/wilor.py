from .file_cache import NpzEstimatorAdapter


class WiLoRAdapter(NpzEstimatorAdapter):
    def __init__(self, root: str) -> None:
        super().__init__("wilor", root)
