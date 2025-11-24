from json import dumps

from pytest import fixture


@fixture
def cache_root(tmp_path):
    return tmp_path


@fixture
def r2_mock_store():
    """
    Provides a mock in-memory R2 store with helpers:
    - store: dict representing the bucket
    - get, put, delete: mock functions for patching
    """
    store = {}

    def mock_get(bucket, key):
        if key in store:
            return store[key], '"etag123"'
        return None, None

    def mock_put(bucket, key, data, acl="public-read", if_match=None):
        # Use canonical JSON for integrity
        if isinstance(data, (dict, list)):
            store[key] = dumps(data, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        elif isinstance(data, bytes):
            store[key] = data
        else:
            store[key] = str(data).encode("utf-8")
        return True

    def mock_delete(bucket, key):
        store.pop(key, None)
        return True

    return store, mock_get, mock_put, mock_delete
