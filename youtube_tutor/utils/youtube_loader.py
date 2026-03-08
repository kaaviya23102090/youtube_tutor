import re
import os


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_video_metadata(video_id: str) -> dict:
    """Fetch video title using YouTube Data API, with safe fallback."""
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key or api_key == "none":
        return {
            "title": f"Video {video_id}",
            "description": "",
            "channel": "Unknown",
            "published_at": "",
        }

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if response["items"]:
            snippet = response["items"][0]["snippet"]
            return {
                "title": snippet.get("title", f"Video {video_id}"),
                "description": snippet.get("description", ""),
                "channel": snippet.get("channelTitle", "Unknown"),
                "published_at": snippet.get("publishedAt", ""),
            }
    except Exception:
        pass

    return {
        "title": f"Video {video_id}",
        "description": "",
        "channel": "Unknown",
        "published_at": "",
    }


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_transcript_with_timestamps(video_id: str) -> list:
    """
    Fetch transcript segments with start timestamps.
    Works with both old and new versions of youtube_transcript_api.
    """
    try:
        # Try new API style first (v0.6+)
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        enriched = []
        for segment in fetched:
            enriched.append({
                "text": segment.text,
                "start": segment.start,
                "duration": segment.duration,
                "timestamp_str": seconds_to_timestamp(segment.start),
            })
        if enriched:
            return enriched
    except Exception as e1:
        pass

    try:
        # Fallback to old API style
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        enriched = []
        for segment in transcript:
            enriched.append({
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment.get("duration", 0),
                "timestamp_str": seconds_to_timestamp(segment["start"]),
            })
        return enriched
    except Exception as e2:
        raise RuntimeError(
            f"Could not fetch transcript for video {video_id}. "
            f"Make sure the video has captions/subtitles enabled. Error: {e2}"
        )


def build_chunks_with_timestamps(
    transcript_segments: list,
    chunk_size: int = 5,
) -> list:
    """
    Merge transcript segments into chunks of chunk_size segments each.
    Each chunk retains the START timestamp of its first segment.
    Returns list of dicts: {text, start_seconds, timestamp_str}
    """
    chunks = []
    for i in range(0, len(transcript_segments), chunk_size):
        group = transcript_segments[i: i + chunk_size]
        combined_text = " ".join(seg["text"] for seg in group)
        chunks.append({
            "text": combined_text,
            "start_seconds": group[0]["start"],
            "timestamp_str": group[0]["timestamp_str"],
        })
    return chunks