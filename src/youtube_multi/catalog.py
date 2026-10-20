"""Enumerate a channel through the official, quota-limited YouTube Data API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from itertools import zip_longest

from yt_dlp import YoutubeDL

from .transcribe import atomic_json
from .fetch import network_options


def _request(resource: str, params: dict, key: str) -> dict:
    query = urllib.parse.urlencode({**params, "key": key})
    try:
        with urllib.request.urlopen(f"https://www.googleapis.com/youtube/v3/{resource}?{query}", timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not leak the API key contained in the URL/error body.
        raise RuntimeError(f"YouTube Data API returned HTTP {exc.code}; check API access and quota") from None
    except urllib.error.URLError:
        raise RuntimeError("YouTube Data API could not be reached") from None


def duration_seconds(value: str) -> int | None:
    match = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return None
    return sum(int(n or 0) * factor for n, factor in zip(match.groups(), [86400, 3600, 60, 1]))


def inventory(channel: str, key: str, *, limit: int | None = None) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    value = channel.strip().rstrip("/")
    if "://" in value:
        parsed = urllib.parse.urlparse(value)
        if parsed.hostname not in {"youtube.com", "www.youtube.com"}:
            raise ValueError("Expected a YouTube channel URL, handle, or UC channel ID")
        parts = parsed.path.strip("/").split("/")
        value = parts[1] if parts[0] == "channel" and len(parts) > 1 else parts[0]
    selector = {"id": value} if value.startswith("UC") else {"forHandle": value}
    response = _request("channels", {"part": "contentDetails,snippet,statistics", **selector}, key)
    channels = response.get("items", [])
    if not channels:
        raise ValueError("Channel not found")
    info = channels[0]
    uploads = info["contentDetails"]["relatedPlaylists"]["uploads"]
    rows: dict[str, dict] = {}
    token = None
    complete = False
    while True:
        params = {"part": "snippet,contentDetails", "playlistId": uploads, "maxResults": 50}
        if token:
            params["pageToken"] = token
        page = _request("playlistItems", params, key)
        for item in page.get("items", []):
            vid = item.get("contentDetails", {}).get("videoId")
            if not vid or vid in rows:
                continue
            snippet = item.get("snippet", {})
            rows[vid] = {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
                         "title": snippet.get("title"), "duration": None, "availability": "unresolved"}
            if limit and len(rows) >= limit:
                break
        token = page.get("nextPageToken")
        # A limited enumeration never makes a completeness claim.
        if limit and len(rows) >= limit:
            break
        if not token:
            complete = True
            break
    ids = list(rows)
    for offset in range(0, len(ids), 50):
        detail = _request("videos", {"part": "snippet,contentDetails,status", "id": ",".join(ids[offset:offset + 50])}, key)
        for video in detail.get("items", []):
            snippet = video.get("snippet", {})
            rows[video["id"]].update({
                "title": snippet.get("title"), "description": snippet.get("description"),
                "channel": snippet.get("channelTitle"), "channel_id": snippet.get("channelId"),
                "published_at": snippet.get("publishedAt"),
                "duration": duration_seconds(video.get("contentDetails", {}).get("duration", "")),
                "availability": video.get("status", {}).get("privacyStatus", "unknown"),
                "caption_flag": video.get("contentDetails", {}).get("caption"),
            })
    videos = list(rows.values())
    for row in videos:
        if row["availability"] == "unresolved":
            row["availability"] = "unavailable"
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "channel_id": info["id"], "channel_title": info.get("snippet", {}).get("title"),
        "uploads_playlist": uploads, "enumeration_complete": complete,
        "reported_public_video_count": info.get("statistics", {}).get("videoCount"),
        "video_count": len(videos), "known_duration_seconds": sum(v["duration"] or 0 for v in videos),
        "unknown_duration_count": sum(v["duration"] is None for v in videos), "videos": videos,
    }


def inventory_ytdlp(channel: str, *, limit: int | None = None) -> dict:
    """Key-free discovery of the videos/shorts/streams tabs; availability is best effort."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    value = channel.strip().rstrip("/")
    if "://" in value:
        parsed = urllib.parse.urlparse(value)
        if parsed.hostname not in {"youtube.com", "www.youtube.com"}:
            raise ValueError("Expected a YouTube channel URL")
        parts = parsed.path.strip("/").split("/")
        value = parts[1] if parts[0] == "channel" and len(parts) > 1 else parts[0]
    if not re.fullmatch(r"(?:@[\w.-]+|UC[\w-]+)", value):
        raise ValueError("Use an @handle or UC channel ID")
    base = f"https://www.youtube.com/{value}" if value.startswith("@") else f"https://www.youtube.com/channel/{value}"
    groups, warnings = [], []
    channel_id = channel_title = None
    for tab in ("videos", "shorts", "streams"):
        options = {**network_options(), "extract_flat": True, "skip_download": True, "quiet": True, "no_warnings": True,
                   "socket_timeout": 20, "retries": 1, "extractor_retries": 1}
        if limit:
            options["playlistend"] = limit
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(f"{base}/{tab}", download=False)
            channel_id = channel_id or info.get("channel_id")
            channel_title = channel_title or info.get("channel")
            group = []
            for entry in info.get("entries", []):
                if not entry or not re.fullmatch(r"[\w-]{11}", entry.get("id", "")):
                    warnings.append(f"{tab}: an entry could not be resolved")
                    continue
                group.append({"video_id": entry["id"], "url": f"https://youtu.be/{entry['id']}",
                              "title": entry.get("title"), "duration": entry.get("duration"),
                              "channel": channel_title, "channel_id": channel_id, "tab": tab,
                              "availability": entry.get("availability") or "unknown"})
            groups.append(group)
        except Exception as exc:
            warnings.append(f"{tab}: {type(exc).__name__}: {exc}")
            groups.append([])
    # Interleave formats so a limited pilot includes Shorts and streams when present.
    rows = {}
    for group in zip_longest(*groups):
        for row in group:
            if row is not None:
                rows.setdefault(row["video_id"], row)
    videos = list(rows.values())[:limit]
    if not videos:
        raise RuntimeError("No videos discovered: " + "; ".join(warnings))
    return {"schema_version": 1, "backend": "yt-dlp", "created_at": datetime.now(timezone.utc).isoformat(),
            "channel_id": channel_id, "channel_title": channel_title, "scope": ["videos", "shorts", "streams"],
            "enumeration_complete": limit is None and not warnings, "warnings": warnings,
            "video_count": len(videos), "known_duration_seconds": sum(v["duration"] or 0 for v in videos),
            "unknown_duration_count": sum(v["duration"] is None for v in videos), "videos": videos}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multi catalog", description=__doc__)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--key-env", default="YOUTUBE_API_KEY", help="Environment variable holding the API key")
    parser.add_argument("--backend", choices=["api", "yt-dlp"], default="api", help="Official API (default), or key-free tab discovery")
    parser.add_argument("--limit", type=int, help="Stop after this many videos; marks the inventory incomplete")
    args = parser.parse_args(argv)
    key = os.environ.get(args.key_env)
    if args.backend == "api" and not key:
        print(f"Set {args.key_env} to a YouTube Data API key. No Cloud trial or paid model API is required.")
        return 2
    try:
        result = inventory(args.channel, key, limit=args.limit) if args.backend == "api" else inventory_ytdlp(args.channel, limit=args.limit)
        atomic_json(args.out, result)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Catalog failed: {exc}")
        return 2
    print(f"Catalog: {result['video_count']} videos, {result['known_duration_seconds'] / 3600:.1f} known hours -> {args.out}")
    return 0
