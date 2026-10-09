"""Tests for scripts.cli (metadata parsing, rendering, ingest/recognize flow).

    python -m pytest tests/test_cli.py -v
"""
from pathlib import Path

import pytest

from scripts.cli import (
    build_parser,
    build_track_id,
    discover_audio_files,
    format_offset,
    format_table,
    ingest_directory,
    main,
    parse_filename,
    plan_tracks,
    render_matches,
    slugify,
)
from tests.conftest import crop, wav_bytes


# ------------------------------------------------------------------ metadata
@pytest.mark.parametrize(
    "stem, expected",
    [
        ("Daft Punk - Around the World", ("Around the World", "Daft Punk")),
        ("01 - Daft Punk - Around the World", ("Around the World", "Daft Punk")),
        ("03. Lonely Song", ("Lonely Song", "Unknown Artist")),
        ("no_separator_here", ("no separator here", "Unknown Artist")),
        # 4 digits is not a track number
        ("1999 - Prince", ("Prince", "1999")),
    ],
)
def test_parse_filename(stem, expected):
    assert parse_filename(stem) == expected


def test_slug_and_track_id():
    assert slugify("Beyoncé & Jay-Z!") == "beyoncé-jay-z"
    assert build_track_id("Daft Punk", "Around the World",
                          "x") == "daft-punk-around-the-world"
    assert build_track_id("Unknown Artist", "Lonely Song",
                          "x") == "lonely-song"
    assert build_track_id("Unknown Artist", "!!!", "!!!").startswith("track-")


def test_discover_audio_files(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["a.mp3", "b.WAV", "c.flac", "d.m4a", "notes.txt", "._hidden.mp3", "sub/e.mp3"]:
        (tmp_path / name).write_bytes(b"x")

    flat = [p.name for p in discover_audio_files(tmp_path)]
    assert flat == ["a.mp3", "b.WAV", "c.flac", "d.m4a"]
    deep = [p.name for p in discover_audio_files(tmp_path, recursive=True)]
    assert "e.mp3" in deep and "._hidden.mp3" not in deep and "notes.txt" not in deep


def test_duplicate_track_ids_are_disambiguated(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    files = [tmp_path / "a" / "Song.mp3", tmp_path / "b" / "Song.mp3"]
    for f in files:
        f.write_bytes(b"x")
    ids = [t.track_id for t in plan_tracks(files, tmp_path)]
    assert ids[0] == "song" and ids[1].startswith("song-") and ids[0] != ids[1]


# ----------------------------------------------------------------- rendering
def test_format_offset():
    assert format_offset(6.0) == "0:06.0"
    assert format_offset(83.0) == "1:23.0"
    assert format_offset(-1) == "0:00.0"


def test_format_table_alignment():
    table = format_table(["#", "Name"], [["1", "Alpha"], [
                         "10", "B"]], right_aligned=frozenset({0}))
    lines = table.splitlines()
    assert lines[0] == " #  Name"
    assert lines[1] == "--  -----"
    assert lines[2] == " 1  Alpha"


def test_render_matches_table_and_empty():
    result = {
        "query_windows": 3,
        "query_duration_seconds": 20.0,
        "matches": [
            {
                "track_id": "artist-song", "title": "Song", "artist": "Artist",
                "confidence": 0.9731, "best_window_score": 0.99, "matched_windows": 3,
                "query_windows": 3, "offset_seconds": 83.0,
            }
        ],
    }
    text = render_matches(result, Path("clip.wav"), 0.5)
    assert "clip.wav" in text and "3 windows" in text
    assert "0.973" in text and "1:23.0" in text and "3/3" in text and "artist-song" in text

    empty = render_matches({**result, "matches": []}, Path("clip.wav"), 0.5)
    assert "No matches above threshold 0.5" in empty


# ------------------------------------------------------------------- parsing
def test_parser_requires_command_and_validates_score():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["ingest-dir", ".", "--ai-generated-score", "2"])
    args = build_parser().parse_args(
        ["recognize", "q.wav", "--limit", "3", "--json"])
    assert args.limit == 3 and args.json and args.threshold == 0.5


def test_dry_run_needs_no_database(tmp_path, capsys):
    (tmp_path / "Artist A - Song A.mp3").write_bytes(b"not real audio")
    assert main(["ingest-dir", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "artist-a-song-a" in out and "Song A" in out and "Dry run" in out


def test_ingest_dir_errors(tmp_path, capsys):
    assert main(["ingest-dir", str(tmp_path / "missing")]) == 2
    assert main(["ingest-dir", str(tmp_path)]) == 1  # empty folder
    assert main(["recognize", str(tmp_path / "nope.wav")]) == 2


# ------------------------------------------------------------------ pipeline
async def test_ingest_then_recognize_flow(matcher, tmp_path, song_a, song_b, capsys):
    (tmp_path / "Artist A - Song A.wav").write_bytes(wav_bytes(song_a))
    (tmp_path / "Artist B - Song B.wav").write_bytes(wav_bytes(song_b))
    (tmp_path / "broken.wav").write_bytes(b"garbage")  # must not stop the run

    tracks = plan_tracks(discover_audio_files(tmp_path), tmp_path)
    summary = await ingest_directory(matcher, tracks, vibe_tags=["test"], ai_generated_score=0.1)

    assert summary.ingested == 2
    assert summary.windows == 22
    assert [p.name for p, _ in summary.failed] == ["broken.wav"]
    assert "[1/3]" in capsys.readouterr().out

    result = await matcher.recognize_audio(wav_bytes(crop(song_a, 6, 16)))
    top = result["matches"][0]
    assert top["track_id"] == "artist-a-song-a"
    assert top["artist"] == "Artist A"
    assert top["vibe_tags"] == ["test"]
    assert top["offset_seconds"] == pytest.approx(6.0, abs=0.01)
    assert "artist-a-song-a" in render_matches(result, Path("clip.wav"), 0.5)


async def test_skip_existing(matcher, tmp_path, song_a, capsys):
    (tmp_path / "Artist A - Song A.wav").write_bytes(wav_bytes(song_a))
    tracks = plan_tracks(discover_audio_files(tmp_path), tmp_path)

    first = await ingest_directory(matcher, tracks, skip_existing=True)
    second = await ingest_directory(matcher, tracks, skip_existing=True)
    assert (first.ingested, first.skipped) == (1, 0)
    assert (second.ingested, second.skipped) == (0, 1)
