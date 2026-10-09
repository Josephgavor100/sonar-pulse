"""API tests: status codes and JSON schemas (fast fake embedder, in-memory Qdrant).

    python -m pytest tests/test_api.py -v
"""
import httpx
import pytest
import pytest_asyncio

from app.api.deps import get_matcher
from app.main import app
from tests.conftest import crop, wav_bytes


@pytest_asyncio.fixture
async def api(matcher):
    app.dependency_overrides[get_matcher] = lambda: matcher
    transport = httpx.ASGITransport(app=app)  # does not run the lifespan (no real Qdrant/model)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def upload(data: bytes, name: str = "clip.wav") -> dict:
    return {"file": (name, data, "audio/wav")}


async def ingest(api, wave, track_id="song_a", **extra):
    form = {"track_id": track_id, "title": f"Title {track_id}", "artist": "Artist", **extra}
    return await api.post("/api/v1/ingest", files=upload(wav_bytes(wave)), data=form)


async def test_health(api):
    assert (await api.get("/health")).json() == {"status": "ok"}


async def test_ingest_success_schema(api, song_a):
    r = await ingest(api, song_a, vibe_tags='["chill", "lofi"]', ai_generated_score="0.1")
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"track_id", "title", "artist", "windows_indexed", "duration_seconds"}
    assert body["track_id"] == "song_a"
    assert body["windows_indexed"] == 11
    assert body["duration_seconds"] == pytest.approx(30.0, abs=0.01)


async def test_ingest_accepts_comma_separated_tags(api, song_a):
    r = await ingest(api, song_a, vibe_tags="chill, lofi")
    assert r.status_code == 201


@pytest.mark.parametrize(
    "form_override, expected",
    [
        ({"ai_generated_score": "1.5"}, 422),  # out of range
        ({"vibe_tags": "[not json"}, 422),
        ({"track_id": ""}, 422),
    ],
)
async def test_ingest_rejects_bad_form(api, song_a, form_override, expected):
    form = {"track_id": "t", "title": "T", "artist": "A", **form_override}
    r = await api.post("/api/v1/ingest", files=upload(wav_bytes(song_a)), data=form)
    assert r.status_code == expected


async def test_ingest_missing_field_is_422(api, song_a):
    r = await api.post(
        "/api/v1/ingest", files=upload(wav_bytes(song_a)), data={"track_id": "t", "title": "T"}
    )
    assert r.status_code == 422


async def test_ingest_missing_file_is_422(api):
    r = await api.post("/api/v1/ingest", data={"track_id": "t", "title": "T", "artist": "A"})
    assert r.status_code == 422


async def test_ingest_empty_file_is_400(api):
    r = await api.post(
        "/api/v1/ingest", files=upload(b""), data={"track_id": "t", "title": "T", "artist": "A"}
    )
    assert r.status_code == 400


async def test_ingest_undecodable_audio_is_422(api):
    r = await api.post(
        "/api/v1/ingest",
        files=upload(b"definitely not audio", "x.wav"),
        data={"track_id": "t", "title": "T", "artist": "A"},
    )
    assert r.status_code == 422


async def test_recognize_success_schema(api, song_a, song_b):
    await ingest(api, song_a, "song_a", vibe_tags='["chill"]', ai_generated_score="0.1")
    await ingest(api, song_b, "song_b")

    r = await api.post("/api/v1/recognize", files=upload(wav_bytes(crop(song_a, 6, 16))))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"matches", "query_windows", "query_duration_seconds"}
    assert body["query_windows"] == 1
    assert len(body["matches"]) == 1

    match = body["matches"][0]
    assert set(match) == {
        "track_id", "title", "artist", "vibe_tags", "ai_generated_score", "confidence",
        "best_window_score", "matched_windows", "query_windows", "offset_seconds",
    }
    assert match["track_id"] == "song_a"
    assert match["vibe_tags"] == ["chill"]
    assert match["ai_generated_score"] == 0.1
    assert match["offset_seconds"] == pytest.approx(6.0, abs=0.01)
    assert 0.9 < match["confidence"] <= 1.0001


async def test_recognize_filters(api, song_a):
    await ingest(api, song_a, "human", vibe_tags='["chill"]', ai_generated_score="0.05")
    await ingest(api, song_a, "synthetic", vibe_tags='["energetic"]', ai_generated_score="0.95")
    clip = upload(wav_bytes(crop(song_a, 0, 10)))

    r = await api.post("/api/v1/recognize", files=clip, data={"max_ai_generated_score": "0.5"})
    assert [m["track_id"] for m in r.json()["matches"]] == ["human"]

    r = await api.post("/api/v1/recognize", files=clip, data={"vibe_tags": "energetic"})
    assert [m["track_id"] for m in r.json()["matches"]] == ["synthetic"]


async def test_recognize_no_match_returns_empty_list(api, song_a, song_b):
    await ingest(api, song_a, "song_a")
    r = await api.post("/api/v1/recognize", files=upload(wav_bytes(crop(song_b, 0, 10))))
    assert r.status_code == 200
    assert r.json()["matches"] == []


async def test_recognize_errors(api):
    assert (await api.post("/api/v1/recognize", files=upload(b""))).status_code == 400
    assert (await api.post("/api/v1/recognize", files=upload(b"nope"))).status_code == 422
    assert (await api.post("/api/v1/recognize")).status_code == 422
    r = await api.post(
        "/api/v1/recognize", files=upload(b"x"), data={"max_ai_generated_score": "2"}
    )
    assert r.status_code == 422