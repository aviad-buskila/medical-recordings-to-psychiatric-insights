"""Data ingestion utilities for pickle loader."""

import pickle
from pathlib import Path
from typing import Any


class DatasetPickleLoader:
    """Loads gold transcripts/casenotes from dataset.pickle."""

    def __init__(self, dataset_pickle_path: str) -> None:
        self.path = Path(dataset_pickle_path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("rb") as f:
            # CR: using pickle is bad practice(secuirty risk), better use json.loads()
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return {}
        return payload

    def load_transcripts(self) -> dict[str, str]:
        payload = self.load()
        transcripts = payload.get("transcripts", {})
        if not isinstance(transcripts, dict):
            return {}
        return {str(k): str(v) for k, v in transcripts.items()}
