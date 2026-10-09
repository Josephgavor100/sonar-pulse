"""Tests for app.services.qdrant_service using Qdrant's in-memory mode.

    python -m pytest tests/test_qdrant_service.py -v
"""
import numpy as np
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models

from app.services.qdrant_service import QdrantService

DIM = 768
COLLECTION = "test_audio"


def unit_basis(i: int) -> list[float]:
    v = np.zeros(DIM)
    v[i] = 1.0
    return v.tolist()


def random_unit(seed: int) -> list[float]:
    v = np.random.default_rng(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def make_point(track_id, window_index, vector, **payload_overrides):
    payload = {
        "track_id": track_id,
        "title": f"Title {track_id}",
        "artist": "Test Artist",
        "vibe_tags": [],
        "ai_generated_score": None,
        "timestamp": window_index * 5.0,
        "window_index": window_index,
    }
    payload.update(payload_overrides)
    return {"vector": vector, "payload": payload}


@pytest_asyncio.fixture
async def client():
    c = AsyncQdrantClient(location=":memory:")
    yield c
    await c.close()


@pytest_asyncio.fixture
async def service(client):
    svc = QdrantService(client, collection_name=COLLECTION)
    await svc.ensure_collection_exists()
    return svc


async def test_collection_initialization(client):
    svc = QdrantService(client, collection_name=COLLECTION)
    assert not await client.collection_exists(COLLECTION)

    await svc.ensure_collection_exists()
    assert await client.collection_exists(COLLECTION)

    params = (await client.get_collection(COLLECTION)).config.params.vectors
    assert params["ast"].size == DIM
    assert params["ast"].distance == models.Distance.COSINE

    await svc.ensure_collection_exists()  # idempotent, must not raise


async def test_upsert_and_search_returns_self_as_best_match(service):
    points = [make_point(f"t{i}", 0, random_unit(i)) for i in range(3)]
    assert await service.upsert_audio_embeddings(points) is True

    results = await service.search_similar_audio(random_unit(1), limit=3, score_threshold=0.0)
    assert results[0]["payload"]["track_id"] == "t1"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert results[0]["payload"]["title"] == "Title t1"
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)


async def test_upsert_is_idempotent(service, client):
    point = make_point("t1", 0, random_unit(1))
    await service.upsert_audio_embeddings([point])
    await service.upsert_audio_embeddings([point])
    assert (await client.count(COLLECTION, exact=True)).count == 1


async def test_score_threshold(service):
    mix = ((np.array(unit_basis(0)) + np.array(unit_basis(1))) / np.sqrt(2)).tolist()
    await service.upsert_audio_embeddings(
        [
            make_point("same", 0, unit_basis(0)),  # cosine 1.0 with query
            make_point("mixed", 0, mix),  # cosine ~0.707
            make_point("orthogonal", 0, unit_basis(1)),  # cosine 0.0
        ]
    )
    query = unit_basis(0)

    ids = lambda rs: {r["payload"]["track_id"] for r in rs}  # noqa: E731
    assert ids(await service.search_similar_audio(query, score_threshold=0.5)) == {"same", "mixed"}
    assert ids(await service.search_similar_audio(query, score_threshold=0.9)) == {"same"}


async def test_payload_filters(service):
    v = random_unit(7)
    await service.upsert_audio_embeddings(
        [
            make_point("human", 0, v, vibe_tags=["chill"], ai_generated_score=0.05),
            make_point("synthetic", 0, v, vibe_tags=["energetic"], ai_generated_score=0.95),
        ]
    )

    async def found(**kw):
        rs = await service.search_similar_audio(v, score_threshold=0.5, **kw)
        return {r["payload"]["track_id"] for r in rs}

    assert await found() == {"human", "synthetic"}
    assert await found(vibe_tags=["chill"]) == {"human"}
    assert await found(max_ai_generated_score=0.5) == {"human"}
    assert await found(track_id="synthetic") == {"synthetic"}
    assert await found(vibe_tags=["energetic"], max_ai_generated_score=0.5) == set()


async def test_delete_points_by_track_id(service, client):
    await service.upsert_audio_embeddings(
        [
            make_point("keep", 0, random_unit(1)),
            make_point("keep", 1, random_unit(2)),
            make_point("drop", 0, random_unit(3)),
            make_point("drop", 1, random_unit(4)),
        ]
    )
    assert (await client.count(COLLECTION, exact=True)).count == 4

    assert await service.delete_points_by_track_id("drop") is True
    assert (await client.count(COLLECTION, exact=True)).count == 2

    results = await service.search_similar_audio(random_unit(3), limit=10, score_threshold=-1.0)
    assert {r["payload"]["track_id"] for r in results} == {"keep"}

    assert await service.delete_points_by_track_id("does-not-exist") is True


async def test_rejects_malformed_input(service):
    with pytest.raises(ValueError):
        await service.upsert_audio_embeddings([make_point("t", 0, [0.1] * 10)])
    with pytest.raises(ValueError):
        await service.upsert_audio_embeddings([{"vector": random_unit(1), "payload": {}}])
    with pytest.raises(ValueError):
        await service.search_similar_audio([0.1] * 10)
    assert await service.upsert_audio_embeddings([]) is True