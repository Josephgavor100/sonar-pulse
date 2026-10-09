"""Regression tests for Qdrant connection routing (path / :memory: / URL / host fallback).

    python -m pytest tests/test_qdrant_connection.py -v
"""
import pytest

from app.core.config import (
    Settings,
    get_settings,
    is_local_location,
    normalize_local_location,
)
from app.db import qdrant as qdrant_db
from app.db.qdrant import create_qdrant_client
from app.services.qdrant_service import QdrantService
from scripts.cli import build_parser, build_settings


class FakeClient:
    """Records how AsyncQdrantClient would have been constructed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(qdrant_db, "AsyncQdrantClient", FakeClient)


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    for var in ("QDRANT_URL", "QDRANT_LOCATION", "QDRANT_HOST", "QDRANT_PORT", "QDRANT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize(
    "value",
    [":memory:", "./qdrant_db", "../db", "/var/lib/qdrant", "path:./db", ".\\qdrant_db",
     "..\\db", "C:\\data\\qdrant", "C:/data/qdrant", "~/qdrant", "  ./padded  "],
)
def test_local_values_detected(value):
    assert is_local_location(value)


@pytest.mark.parametrize("value", ["http://localhost:6333", "https://x.cloud.qdrant.io:6333", "localhost:6333", "", None])
def test_remote_values_not_local(value):
    assert not is_local_location(value)


def test_normalize_strips_path_prefix():
    assert normalize_local_location(" path:./db ") == "./db"
    assert normalize_local_location(":memory:") == ":memory:"


# ----------------------------------------------------------------- settings
def test_default_host_is_localhost():
    assert get_settings().qdrant_host == "localhost"


@pytest.mark.parametrize("raw, expected", [(":memory:", ":memory:"), ("./qdrant_db", "./qdrant_db"), ("path:../db", "../db")])
def test_env_url_that_is_a_path_becomes_location(monkeypatch, raw, expected):
    monkeypatch.setenv("QDRANT_URL", raw)
    s = get_settings()
    assert s.qdrant_location == expected
    assert s.qdrant_url is None


def test_env_remote_url_stays_url(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    s = get_settings()
    assert s.qdrant_url == "http://qdrant:6333" and s.qdrant_location is None


# ------------------------------------------------------------------- client
def test_memory_location(fake_client):
    c = create_qdrant_client(Settings(qdrant_location=":memory:"))
    assert c.kwargs == {"location": ":memory:"}


@pytest.mark.parametrize("loc", ["./qdrant_db", "path:./qdrant_db"])
def test_folder_location_uses_path_not_location(fake_client, loc):
    c = create_qdrant_client(Settings(qdrant_location=loc))
    assert c.kwargs == {"path": "qdrant_db"}  # Path() normalizes "./"


def test_path_inside_url_field_is_treated_as_local(fake_client):
    c = create_qdrant_client(
        Settings(qdrant_url="./qdrant_db", qdrant_host="qdrant"))
    assert "path" in c.kwargs and "url" not in c.kwargs and "host" not in c.kwargs


def test_memory_inside_url_field_is_treated_as_local(fake_client):
    assert create_qdrant_client(Settings(qdrant_url=":memory:")).kwargs == {
        "location": ":memory:"}


def test_remote_url(fake_client):
    c = create_qdrant_client(
        Settings(qdrant_url="https://x:6333", qdrant_api_key="k"))
    assert c.kwargs == {"url": "https://x:6333", "api_key": "k"}


def test_location_wins_over_url(fake_client):
    c = create_qdrant_client(
        Settings(qdrant_location=":memory:", qdrant_url="http://x:6333"))
    assert c.kwargs == {"location": ":memory:"}


def test_host_port(fake_client, monkeypatch):
    monkeypatch.setattr(qdrant_db, "_host_resolves",
                        lambda h: pytest.fail("no DNS check for normal hosts"))
    c = create_qdrant_client(
        Settings(qdrant_host="localhost", qdrant_port=6400))
    assert c.kwargs == {"host": "localhost", "port": 6400, "api_key": None}


def test_unresolvable_docker_host_falls_back_to_local_storage(fake_client, monkeypatch):
    monkeypatch.setattr(qdrant_db, "_host_resolves", lambda h: False)
    c = create_qdrant_client(Settings(qdrant_host="qdrant"))
    assert c.kwargs == {"path": "qdrant_db"}


def test_resolvable_docker_host_is_used_remotely(fake_client, monkeypatch):
    monkeypatch.setattr(qdrant_db, "_host_resolves", lambda h: True)
    c = create_qdrant_client(Settings(qdrant_host="qdrant"))
    assert c.kwargs["host"] == "qdrant"


# ---------------------------------------------------------------------- CLI
@pytest.mark.parametrize(
    "arg, location, url",
    [
        ("./qdrant_db", "./qdrant_db", None),
        ("path:./qdrant_db", "./qdrant_db", None),
        (":memory:", ":memory:", None),
        ("C:\\data\\q", "C:\\data\\q", None),
        ("http://localhost:6333", None, "http://localhost:6333"),
    ],
)
def test_cli_build_settings_routes_targets(monkeypatch, arg, location, url):
    # the stale value from the original bug
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    get_settings.cache_clear()
    args = build_parser().parse_args(
        ["recognize", "q.wav", "--qdrant-url", arg, "--collection", "c1"])
    s = build_settings(args)
    assert (s.qdrant_location, s.qdrant_url,
            s.qdrant_collection) == (location, url, "c1")


def test_cli_without_flag_keeps_env_settings(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "./from_env")
    get_settings.cache_clear()
    s = build_settings(build_parser().parse_args(["recognize", "q.wav"]))
    assert s.qdrant_location == "./from_env" and s.qdrant_url is None


# ---------------------------------------------- real local storage (no fake)
async def test_folder_storage_persists_between_clients(tmp_path):
    settings = Settings(qdrant_location=str(tmp_path / "db"))

    first = create_qdrant_client(settings)
    await QdrantService(first, collection_name="persist_test").ensure_collection_exists()
    await first.close()

    second = create_qdrant_client(settings)
    assert await second.collection_exists("persist_test")
    await second.close()
