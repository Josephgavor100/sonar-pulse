"""Tests for app.services.matcher.

Most tests use a fast fake embedder; the one marked `slow` uses the real AST
model.   python -m pytest tests/test_matcher.py -v      (add -m "not slow" to skip it)
"""
import pytest

from app.services.audio_processor import AudioProcessingError
from app.services.matcher import AudioMatcher, score_track_matches
from tests.conftest import COLLECTION, SONG_A_FREQS, crop, make_song, meta, wav_bytes


# ------------------------------------------------------------- pure scoring
def hit(track_id: str, timestamp: float, score: float) -> dict:
    return {"score": score, "payload": {"track_id": track_id, "title": track_id, "timestamp": timestamp}}


def test_offset_consistency_beats_a_single_strong_hit():
    query_starts = [0.0, 5.0, 10.0]
    window_hits = [
        [hit("consistent", 30.0, 0.80), hit("scattered", 10.0, 0.95)],
        [hit("consistent", 35.0, 0.80), hit("scattered", 100.0, 0.60)],
        [hit("consistent", 40.0, 0.80), hit("scattered", 7.0, 0.60)],
    ]
    matches = score_track_matches(window_hits, query_starts, tolerance=2.0)

    assert [m["track_id"] for m in matches] == ["consistent", "scattered"]
    top, other = matches
    assert top["matched_windows"] == 3
    assert top["confidence"] == pytest.approx(0.80)
    assert top["offset_seconds"] == pytest.approx(30.0)
    assert other["matched_windows"] == 1  # only its best alignment counts
    assert other["confidence"] == pytest.approx(0.95 / 3)


def test_tolerance_absorbs_grid_jitter():
    # Stored windows sit on a 2 s grid, so deltas wobble by up to +/-1 s.
    window_hits = [[hit("t", 29.0, 0.9)], [hit("t", 36.0, 0.9)], [hit("t", 40.0, 0.9)]]
    (match,) = score_track_matches(window_hits, [0.0, 5.0, 10.0], tolerance=2.0)
    assert match["matched_windows"] == 3
    assert match["offset_seconds"] == pytest.approx(30.0)


def test_no_hits_gives_no_matches():
    assert score_track_matches([[], []], [0.0, 5.0], tolerance=2.0) == []


# ---------------------------------------------------------------- ingestion
async def test_ingest_builds_windows_with_metadata(matcher, qdrant_client, song_a):
    result = await matcher.ingest_track(
        wav_bytes(song_a), meta("song_a", vibe_tags=["chill"], ai_generated_score=0.1)
    )
    # 30 s, 10 s window, 2 s hop -> starts 0, 2, ..., 20
    assert result["windows_indexed"] == 11
    assert result["duration_seconds"] == pytest.approx(30.0, abs=0.01)

    points, _ = await qdrant_client.scroll(COLLECTION, limit=100, with_payload=True)
    payloads = sorted((p.payload for p in points), key=lambda p: p["window_index"])
    assert [p["window_index"] for p in payloads] == list(range(11))
    assert [p["timestamp"] for p in payloads] == [2.0 * i for i in range(11)]
    assert payloads[0]["track_id"] == "song_a"
    assert payloads[0]["vibe_tags"] == ["chill"]
    assert payloads[0]["ai_generated_score"] == 0.1
    assert {str(p.id) for p in points} == {matcher.qdrant.make_point_id("song_a", i) for i in range(11)}


async def test_reingest_replaces_old_windows(matcher, qdrant_client, song_a):
    await matcher.ingest_track(wav_bytes(song_a), meta("song_a"))
    await matcher.ingest_track(wav_bytes(crop(song_a, 0, 20)), meta("song_a"))
    # 20 s -> starts 0, 2, ..., 10 -> 6 windows; no stale leftovers from the 30 s version
    assert (await qdrant_client.count(COLLECTION, exact=True)).count == 6


async def test_ingest_accepts_file_path(matcher, tmp_path, song_a):
    path = tmp_path / "a.wav"
    path.write_bytes(wav_bytes(song_a))
    result = await matcher.ingest_track(path, meta("from_path"))
    assert result["windows_indexed"] == 11


async def test_ingest_validates_input(matcher, song_a):
    audio = wav_bytes(song_a)
    with pytest.raises(ValueError):
        await matcher.ingest_track(audio, {"track_id": "x", "artist": "a"})  # no title
    with pytest.raises(ValueError):
        await matcher.ingest_track(audio, meta("x", ai_generated_score=1.5))
    with pytest.raises(ValueError):
        await matcher.ingest_track(audio, meta("x", vibe_tags="chill"))
    with pytest.raises(AudioProcessingError):
        await matcher.ingest_track(b"not audio", meta("x"))


# -------------------------------------------------------------- recognition
async def test_recognizes_exact_excerpt_with_offset(matcher, song_a, song_b):
    await matcher.ingest_track(wav_bytes(song_a), meta("song_a"))
    await matcher.ingest_track(wav_bytes(song_b), meta("song_b"))

    result = await matcher.recognize_audio(wav_bytes(crop(song_a, 6, 16)))

    assert result["query_windows"] == 1
    assert result["query_duration_seconds"] == pytest.approx(10.0, abs=0.01)
    assert [m["track_id"] for m in result["matches"]] == ["song_a"]  # song_b is far below threshold
    top = result["matches"][0]
    assert top["title"] == "Title song_a"
    assert top["confidence"] > 0.95
    assert top["offset_seconds"] == pytest.approx(6.0, abs=0.01)


async def test_recognizes_multi_window_query(matcher, song_a, song_b):
    await matcher.ingest_track(wav_bytes(song_a), meta("song_a"))
    await matcher.ingest_track(wav_bytes(song_b), meta("song_b"))

    # 20 s query -> windows at 0, 5, 10 s; the 5 s one is off the 2 s grid
    result = await matcher.recognize_audio(wav_bytes(crop(song_a, 4, 24)))

    top = result["matches"][0]
    assert top["track_id"] == "song_a"
    assert top["query_windows"] == 3
    assert top["matched_windows"] == 3
    assert top["confidence"] > 0.8
    assert top["offset_seconds"] == pytest.approx(4.0, abs=1.0)


async def test_unrelated_audio_returns_no_matches(matcher, song_a, song_b):
    await matcher.ingest_track(wav_bytes(song_a), meta("song_a"))
    result = await matcher.recognize_audio(wav_bytes(crop(song_b, 0, 10)))
    assert result["matches"] == []


async def test_recognize_filters_and_limit(matcher, song_a):
    audio = wav_bytes(song_a)
    await matcher.ingest_track(audio, meta("human", vibe_tags=["chill"], ai_generated_score=0.05))
    await matcher.ingest_track(audio, meta("synthetic", vibe_tags=["energetic"], ai_generated_score=0.95))
    query = wav_bytes(crop(song_a, 0, 10))

    both = await matcher.recognize_audio(query)
    assert {m["track_id"] for m in both["matches"]} == {"human", "synthetic"}

    only_human = await matcher.recognize_audio(query, max_ai_generated_score=0.5)
    assert [m["track_id"] for m in only_human["matches"]] == ["human"]

    energetic = await matcher.recognize_audio(query, vibe_tags=["energetic"])
    assert [m["track_id"] for m in energetic["matches"]] == ["synthetic"]

    assert len((await matcher.recognize_audio(query, limit=1))["matches"]) == 1
    with pytest.raises(ValueError):
        await matcher.recognize_audio(query, limit=0)


# ------------------------------------------------------ real model (slow)
@pytest.mark.slow
async def test_end_to_end_with_real_ast_embedder(qdrant_client):
    from app.services.embedder import AudioEmbedder
    from app.services.qdrant_service import QdrantService

    svc = QdrantService(qdrant_client, collection_name="real_model")
    await svc.ensure_collection_exists()
    real = AudioMatcher(embedder=AudioEmbedder(), qdrant=svc)

    song = make_song(SONG_A_FREQS[:10])  # 20 s
    await real.ingest_track(wav_bytes(song), meta("real_song"))
    result = await real.recognize_audio(wav_bytes(crop(song, 4, 14)), score_threshold=0.0)

    top = result["matches"][0]
    assert top["track_id"] == "real_song"
    assert top["offset_seconds"] == pytest.approx(4.0, abs=2.0)