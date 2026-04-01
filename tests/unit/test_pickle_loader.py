import pickle

from src.ingestion.pickle_loader import DatasetPickleLoader


def test_load_transcripts_from_pickle(tmp_path) -> None:
    payload = {"transcripts": {"S1": "hello world"}, "casenotes": {}}
    target = tmp_path / "dataset.pickle"
    with target.open("wb") as f:
        pickle.dump(payload, f, protocol=4)

    loader = DatasetPickleLoader(str(target))
    transcripts = loader.load_transcripts()
    assert transcripts["S1"] == "hello world"
