from orchard import store


def test_add_blob_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "BLOB_DIR", tmp_path)
    sha = store.add_blob(b"hello world")
    assert store.has_blob(sha)
    same = store.add_blob(b"hello world")
    assert same == sha
    assert len(list(tmp_path.rglob("*"))) == 2


def test_blob_path_sharding(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "BLOB_DIR", tmp_path)
    sha = store.sha256_bytes(b"x")
    path = store.blob_path(sha)
    assert path.parent.name == sha[:2]
    assert path.name == sha
