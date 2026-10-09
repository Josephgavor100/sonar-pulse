#!/usr/bin/env python
"""Sonar Pulse command-line tool: bulk-ingest a music folder, recognize clips.

Run from the backend/ directory (or inside the backend container):

    python -m scripts.cli ingest-dir ../data/music --recursive
    python -m scripts.cli recognize ../data/queries/clip.wav

Where the vectors live is chosen with --qdrant-url (or the QDRANT_* environment
variables). It accepts a server URL, a local folder, or ``:memory:``:

    --qdrant-url http://localhost:6333     Qdrant server (docker compose up -d qdrant)
    --qdrant-url ./qdrant_db               local on-disk storage, no Docker needed
    --qdrant-url :memory:                  throwaway, vanishes when the command exits

With no configuration it uses http://localhost:6333.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Sequence, TextIO

# Allow `python scripts/cli.py` as well as `python -m scripts.cli`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import (  # noqa: E402  (lightweight: stdlib only)
    MEMORY_LOCATION,
    Settings,
    get_settings,
    is_local_location,
    normalize_local_location,
)

if TYPE_CHECKING:  # heavy modules are imported lazily inside commands
    from app.services.matcher import AudioMatcher

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")
UNKNOWN_ARTIST = "Unknown Artist"
MAX_CELL_WIDTH = 34

EPILOG = """\
examples:
  python -m scripts.cli ingest-dir ../data/music --recursive
  python -m scripts.cli ingest-dir ./songs --qdrant-url ./qdrant_db      # local storage, no Docker
  python -m scripts.cli ingest-dir ./songs --dry-run                     # preview metadata only
  python -m scripts.cli ingest-dir ./songs --skip-existing               # resume an interrupted run
  python -m scripts.cli recognize ./clip.wav --qdrant-url ./qdrant_db --limit 3 --threshold 0.6
  python -m scripts.cli recognize ./clip.wav --json

metadata: ID3/Vorbis/MP4 tags are used when `mutagen` is installed; otherwise
(or when tags are missing) the filename is parsed as "Artist - Title.ext".

note: a local storage folder can be opened by only one process at a time, so
stop the API server before using the same folder from the CLI (and vice versa).
"""


# ============================================================ metadata & files
@dataclass(frozen=True)
class TrackInfo:
    path: Path
    track_id: str
    title: str
    artist: str


def discover_audio_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Sorted audio files in ``folder`` (hidden files such as macOS ``._x.mp3`` skipped)."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith(".")
    )


def parse_filename(stem: str) -> tuple[str, str]:
    """Return ``(title, artist)`` from a filename stem.

    ``"01 - Daft Punk - Around the World"`` -> ("Around the World", "Daft Punk");
    ``"03. Lonely Song"`` -> ("Lonely Song", "Unknown Artist").
    """
    cleaned = re.sub(r"^\s*\d{1,3}\s*[-._)]\s*", "", stem)  # leading track number
    cleaned = cleaned.replace("_", " ").strip() or stem.strip()
    if " - " in cleaned:
        artist, title = (part.strip() for part in cleaned.split(" - ", 1))
        if artist and title:
            return title, artist
    return cleaned, UNKNOWN_ARTIST


def read_tags(path: Path) -> tuple[str | None, str | None]:
    """Best-effort ``(title, artist)`` from embedded tags; ``(None, None)`` if unavailable."""
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None, None
    try:
        audio = MutagenFile(path, easy=True)
        tags = getattr(audio, "tags", None)
        if not tags:
            return None, None

        def first(key: str) -> str | None:
            value = tags.get(key)
            text = str(value[0]).strip() if value else ""
            return text or None

        return first("title"), first("artist")
    except Exception:  # corrupt/unsupported tags must never stop a bulk run
        return None, None


def slugify(text: str) -> str:
    return re.sub(r"[\W_]+", "-", text.casefold()).strip("-")


def build_track_id(artist: str, title: str, stem: str) -> str:
    """Stable, readable ID: slug of 'artist title' (or just the title)."""
    base = title if artist == UNKNOWN_ARTIST else f"{artist} {title}"
    slug = slugify(base)[:100].strip("-") or slugify(stem)[:100].strip("-")
    return slug or "track-" + hashlib.sha1(stem.encode()).hexdigest()[:8]


def plan_tracks(files: Sequence[Path], root: Path) -> list[TrackInfo]:
    """Resolve metadata and unique track IDs for each file."""
    seen: set[str] = set()
    tracks: list[TrackInfo] = []
    for path in files:
        tag_title, tag_artist = read_tags(path)
        file_title, file_artist = parse_filename(path.stem)
        title, artist = tag_title or file_title, tag_artist or file_artist
        track_id = build_track_id(artist, title, path.stem)
        if track_id in seen:  # same artist/title in two files: disambiguate by path
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            track_id = f"{track_id}-{hashlib.sha1(str(rel).encode()).hexdigest()[:6]}"
        seen.add(track_id)
        tracks.append(TrackInfo(path=path, track_id=track_id, title=title, artist=artist))
    return tracks


# ================================================================= rendering
def truncate(text: str, width: int = MAX_CELL_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_aligned: frozenset[int] = frozenset(),
) -> str:
    """Plain-text table with a header rule (ASCII only, safe on any console)."""
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        parts = [
            c.rjust(widths[i]) if i in right_aligned else c.ljust(widths[i])
            for i, c in enumerate(cells)
        ]
        return "  ".join(parts).rstrip()

    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)])


def format_offset(seconds: float) -> str:
    """Seconds -> ``m:ss.s`` (e.g. 83.0 -> ``1:23.0``)."""
    minutes, secs = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def render_matches(result: dict, source: Path, threshold: float) -> str:
    """Human-readable report for a ``recognize_audio`` result."""
    windows = result["query_windows"]
    header = (
        f"Query: {source.name}  |  {result['query_duration_seconds']:.1f}s  |  "
        f"{windows} window{'s' if windows != 1 else ''}"
    )
    matches = result["matches"]
    if not matches:
        return f"{header}\n\nNo matches above threshold {threshold}. Try lowering --threshold."

    rows = [
        [
            str(rank),
            f"{m['confidence']:.3f}",
            format_offset(m["offset_seconds"]),
            f"{m['matched_windows']}/{m['query_windows']}",
            f"{m['best_window_score']:.3f}",
            truncate(m["title"] or "?"),
            truncate(m["artist"] or "?"),
            truncate(m["track_id"]),
        ]
        for rank, m in enumerate(matches, start=1)
    ]
    table = format_table(
        ["#", "Confidence", "Offset", "Windows", "Best", "Title", "Artist", "Track ID"],
        rows,
        right_aligned=frozenset({0, 1, 2, 3, 4}),
    )
    return f"{header}\n\n{table}"


# ================================================================= ingestion
@dataclass
class IngestSummary:
    ingested: int = 0
    skipped: int = 0
    windows: int = 0
    seconds: float = 0.0
    failed: list[tuple[Path, str]] = field(default_factory=list)


async def _track_exists(matcher: "AudioMatcher", track_id: str) -> bool:
    from qdrant_client import models

    result = await matcher.qdrant.client.count(
        collection_name=matcher.qdrant.collection_name,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="track_id", match=models.MatchValue(value=track_id))]
        ),
        exact=True,
    )
    return result.count > 0


async def ingest_directory(
    matcher: "AudioMatcher",
    tracks: Sequence[TrackInfo],
    vibe_tags: list[str] | None = None,
    ai_generated_score: float | None = None,
    skip_existing: bool = False,
    out: TextIO | None = None,
) -> IngestSummary:
    """Ingest tracks one by one, reporting progress; one bad file never stops the run."""
    out = out or sys.stdout
    summary = IngestSummary()
    started = time.perf_counter()

    for i, track in enumerate(tracks, start=1):
        label = f"[{i}/{len(tracks)}] {track.artist} - {track.title}"
        try:
            if skip_existing and await _track_exists(matcher, track.track_id):
                summary.skipped += 1
                print(f"{label}\n  skipped (already indexed)", file=out, flush=True)
                continue

            print(label, file=out, flush=True)
            t0 = time.perf_counter()
            result = await matcher.ingest_track(
                track.path,
                {
                    "track_id": track.track_id,
                    "title": track.title,
                    "artist": track.artist,
                    "vibe_tags": vibe_tags or [],
                    "ai_generated_score": ai_generated_score,
                    "source_file": track.path.name,
                },
            )
            summary.ingested += 1
            summary.windows += result["windows_indexed"]
            print(
                f"  -> {result['windows_indexed']} windows "
                f"({result['duration_seconds']:.1f}s audio) in {time.perf_counter() - t0:.1f}s",
                file=out,
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            summary.failed.append((track.path, str(exc)))
            print(f"  ! failed: {exc}", file=out, flush=True)

    summary.seconds = time.perf_counter() - started
    return summary


# ===================================================================== runtime
def build_settings(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides to the environment settings.

    ``--qdrant-url`` may be a server URL, a local folder (``./qdrant_db``,
    ``path:./qdrant_db``) or ``:memory:``. Local targets go to ``qdrant_location``
    and clear ``qdrant_url``; real URLs go to ``qdrant_url`` and clear
    ``qdrant_location``.
    """
    updates: dict = {}
    if args.qdrant_url:
        url = args.qdrant_url.strip()
        if is_local_location(url):
            updates["qdrant_location"] = normalize_local_location(url)
            updates["qdrant_url"] = None
        else:
            updates["qdrant_url"] = url
            updates["qdrant_location"] = None
    if args.collection:
        updates["qdrant_collection"] = args.collection
    return dataclasses.replace(get_settings(), **updates)


@contextlib.asynccontextmanager
async def open_matcher(settings: Settings) -> AsyncIterator["AudioMatcher"]:
    """Connect to Qdrant, ensure the collection, load the model, build a matcher."""
    from app.db.qdrant import create_qdrant_client
    from app.services.embedder import get_embedder
    from app.services.matcher import AudioMatcher
    from app.services.qdrant_service import QdrantService

    location = settings.qdrant_location
    if location == MEMORY_LOCATION:
        print(
            "warning: using :memory: - the database vanishes when this command exits, "
            "so ingest-dir and recognize cannot share data. Use a folder such as ./qdrant_db.",
            file=sys.stderr,
        )
    elif location and is_local_location(location):
        print(f"Using local Qdrant storage at {Path(location).expanduser().resolve()}", file=sys.stderr)

    client = create_qdrant_client(settings)
    try:
        service = QdrantService(
            client,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dim,
        )
        await service.ensure_collection_exists()
        print("Loading embedding model (first run downloads ~350 MB)...", file=sys.stderr, flush=True)
        embedder = await asyncio.to_thread(get_embedder)
        yield AudioMatcher(embedder=embedder, qdrant=service)
    finally:
        await client.close()


def parse_tags_arg(raw: str | None) -> list[str] | None:
    tags = [t.strip() for t in (raw or "").split(",") if t.strip()]
    return tags or None


async def cmd_ingest_dir(args: argparse.Namespace) -> int:
    folder: Path = args.folder
    if not folder.is_dir():
        print(f"error: '{folder}' is not a directory.", file=sys.stderr)
        return 2

    files = discover_audio_files(folder, args.recursive)
    if not files:
        print(
            f"No {'/'.join(AUDIO_EXTENSIONS)} files found in '{folder}'"
            + ("" if args.recursive else " (use --recursive to search subfolders)."),
            file=sys.stderr,
        )
        return 1
    tracks = plan_tracks(files, folder)

    if args.dry_run:
        rows = [[str(t.path.relative_to(folder)), t.track_id, t.title, t.artist] for t in tracks]
        print(format_table(["File", "Track ID", "Title", "Artist"], [[truncate(c, 50) for c in r] for r in rows]))
        print(f"\n{len(tracks)} file(s). Dry run: nothing was ingested.")
        return 0

    async with open_matcher(build_settings(args)) as matcher:
        summary = await ingest_directory(
            matcher,
            tracks,
            vibe_tags=parse_tags_arg(args.vibe_tags),
            ai_generated_score=args.ai_generated_score,
            skip_existing=args.skip_existing,
        )

    print(
        f"\nDone: {summary.ingested} ingested, {summary.skipped} skipped, "
        f"{len(summary.failed)} failed | {summary.windows} windows | {summary.seconds:.1f}s"
    )
    for path, reason in summary.failed:
        print(f"  FAILED {path.name}: {reason}", file=sys.stderr)
    return 1 if summary.failed else 0


async def cmd_recognize(args: argparse.Namespace) -> int:
    if not args.file.is_file():
        print(f"error: '{args.file}' is not a file.", file=sys.stderr)
        return 2

    async with open_matcher(build_settings(args)) as matcher:
        t0 = time.perf_counter()
        result = await matcher.recognize_audio(
            args.file,
            limit=args.limit,
            score_threshold=args.threshold,
            max_ai_generated_score=args.max_ai_score,
            vibe_tags=parse_tags_arg(args.vibe_tags),
        )
        elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_matches(result, args.file, args.threshold))
        print(f"\n({elapsed:.1f}s)")
    return 0


# ======================================================================== CLI
def unit_float(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--qdrant-url",
        metavar="URL_OR_PATH",
        help="Qdrant server URL (http://localhost:6333), local storage folder (./qdrant_db) "
        "or :memory:. Default: QDRANT_* env settings, else http://localhost:6333",
    )
    common.add_argument("--collection", help="Collection name (default: sonar_pulse_audio)")
    common.add_argument("-v", "--verbose", action="store_true", help="Show INFO logs")

    parser = argparse.ArgumentParser(
        prog="sonar-pulse",
        description="Bulk-ingest audio and recognize clips with Sonar Pulse.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{ingest-dir,recognize}")

    ing = sub.add_parser(
        "ingest-dir", parents=[common], help="Index every audio file in a folder",
        description="Index every .mp3/.wav/.flac/.m4a file in a folder. Re-ingesting a track replaces it.",
    )
    ing.add_argument("folder", type=Path, help="Folder containing audio files")
    ing.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    ing.add_argument("--vibe-tags", help="Comma-separated tags applied to every track, e.g. chill,lofi")
    ing.add_argument("--ai-generated-score", type=unit_float, help="0-1 score applied to every track")
    ing.add_argument("--skip-existing", action="store_true", help="Skip tracks whose ID is already indexed")
    ing.add_argument("--dry-run", action="store_true", help="Show detected metadata; ingest nothing")

    rec = sub.add_parser(
        "recognize", parents=[common], help="Identify a query clip",
        description="Match a query clip (e.g. a 10 s recording) against the indexed tracks.",
    )
    rec.add_argument("file", type=Path, help="Query audio file")
    rec.add_argument("--limit", type=int, default=5, help="Max tracks to show (default: 5)")
    rec.add_argument("--threshold", type=float, default=0.5, help="Min per-window similarity (default: 0.5)")
    rec.add_argument("--max-ai-score", type=unit_float, help="Exclude tracks with ai_generated_score above this")
    rec.add_argument("--vibe-tags", help="Only tracks having at least one of these comma-separated tags")
    rec.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for stream in (sys.stdout, sys.stderr):  # never crash on exotic characters in titles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    command = cmd_ingest_dir if args.command == "ingest-dir" else cmd_recognize
    try:
        return asyncio.run(command(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        if "already accessed by another instance" in str(exc):
            print(
                "error: the local Qdrant storage folder is in use by another process "
                "(is the API server running with the same folder?). Stop it, or use a different folder.",
                file=sys.stderr,
            )
            return 1
        raise
    except Exception as exc:  # noqa: BLE001
        from app.services.qdrant_service import QdrantServiceError

        if isinstance(exc, QdrantServiceError):
            print(
                f"error: could not use Qdrant ({exc}).\n"
                "Is the server running?  docker compose up -d qdrant\n"
                "No Docker? Use local storage instead:  --qdrant-url ./qdrant_db",
                file=sys.stderr,
            )
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())