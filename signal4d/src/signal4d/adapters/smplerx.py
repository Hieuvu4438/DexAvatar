from .file_cache import NpzEstimatorAdapter


class SMPLerXAdapter(NpzEstimatorAdapter):
    def __init__(self, root: str) -> None:
        super().__init__("smplerx", root)
