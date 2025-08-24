import os
import json
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import re
import streamlit as st

# ------------------------- Date/Time utilities -------------------------

def parse_youtube_date(date_str: str) -> Optional[datetime]:
    """Parse YouTube's ISO 8601 date format to datetime object."""
    try:
        # YouTube API returns dates in ISO 8601 format like "2024-08-15T10:30:00Z"
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        return datetime.fromisoformat(date_str).astimezone(timezone.utc)
    except Exception:
        return None


def is_within_timeframe(published_date: str, days: Optional[int]) -> bool:
    """Check if video was published within the specified number of days."""
    if not days:
        return True
    
    parsed_date = parse_youtube_date(published_date)
    if not parsed_date:
        return True  # Include if can't parse date
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    return parsed_date >= cutoff_date


# ------------------------- Atomic functions -------------------------

def build_youtube_client(api_key: str):
    """Build a YouTube Data API v3 client."""
    return build("youtube", "v3", developerKey=api_key)


def get_channel_id(youtube, channel_name: str) -> str:
    """Resolve a channel name to channelId using YouTube Data API."""
    try:
        resp = (
            youtube.search()
            .list(q=channel_name, type="channel", part="snippet", maxResults=1)
            .execute()
        )
        items = resp.get("items", [])
        return items[0]["id"]["channelId"] if items else ""
    except Exception:
        return ""


def search_channel_videos(
    youtube, 
    channel_id: str, 
    topic: Optional[str], 
    max_results: int,
    days_filter: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Search videos on a channel, optionally filtered by topic and time."""
    try:
        # Build search parameters
        search_params = {
            "channelId": channel_id,
            "part": "snippet",
            "type": "video",
            "maxResults": min(max_results * 2, 50),  # Get more to filter by date
            "order": "date",
        }
        
        if topic:
            search_params["q"] = topic
        
        # Add time filter if specified
        if days_filter:
            published_after = datetime.now(timezone.utc) - timedelta(days=days_filter)
            search_params["publishedAfter"] = published_after.isoformat()
        
        req = youtube.search().list(**search_params)
        items = req.execute().get("items", [])
        
        videos = []
        for item in items:
            video_data = {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "published_at": item["snippet"].get("publishedAt", ""),
                "description": item["snippet"].get("description", ""),
                "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url", ""),
                "channel_title": item["snippet"].get("channelTitle", ""),
                "tags": item["snippet"].get("tags", [])
            }
            
            # Double-check time filter (API filter might not be perfect)
            if is_within_timeframe(video_data["published_at"], days_filter):
                videos.append(video_data)
            
            # Stop when we have enough videos
            if len(videos) >= max_results:
                break
        
        return videos
        
    except Exception:
        return []


def get_latest_videos(youtube, channel_id: str, max_results: int = 50) -> List[str]:
    """Get IDs of latest videos from channel for relevance matching."""
    try:
        req = youtube.search().list(
            channelId=channel_id,
            part="snippet",
            type="video",
            maxResults=max_results,
            order="date"
        )
        items = req.execute().get("items", [])
        return [item["id"]["videoId"] for item in items]
    except Exception:
        return []


def find_common_relevant_videos(
    topic_videos: List[Dict[str, Any]], 
    latest_video_ids: List[str]
) -> List[Dict[str, Any]]:
    """Find videos that appear in both topic search and latest videos (common/relevant)."""
    topic_video_ids = {v["video_id"] for v in topic_videos}
    latest_video_ids_set = set(latest_video_ids)
    
    # Find intersection (common video IDs)
    common_ids = topic_video_ids.intersection(latest_video_ids_set)
    
    # Return videos that are in both lists
    common_videos = [v for v in topic_videos if v["video_id"] in common_ids]
    
    return common_videos


def fetch_transcript_iterable(video_id: str, languages: Optional[List[str]] = None) -> str:
    """Fetch transcript for a video and return as concatenated text."""
    try:
        # Try to get transcript
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)

# is iterable
        result=""
        for snippet in fetched_transcript:
            result+=snippet.text

# indexable

        
        return result
        
    except Exception:
        return ""


def save_to_json(data: Dict[str, Any], filename: str = "youtube_timeframe.json") -> None:
    """Save data to JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ------------------------- Enhanced Orchestrator function -------------------------

def get_videos_and_transcripts_with_timeframe(
    channel_name: str, 
    topic: Optional[str], 
    days_filter: Optional[int] = None,
    max_results: int = 5,
    find_common: bool = True
) -> Dict[str, Any]:
    """
    Enhanced orchestrator with time filtering and relevance matching:
    
    1) Build YT client
    2) Resolve channel name -> channelId
    3) Find videos matching topic within timeframe
    4) Optionally find common videos between topic search and latest videos
    5) For each relevant video, fetch transcript
    6) Return structured data with filtering metadata
    
    Args:
        channel_name: YouTube channel name
        topic: Search topic/keyword (optional)
        days_filter: Only include videos from last N days (optional)
        max_results: Maximum number of videos to process
        find_common: Whether to find intersection of topic + latest videos
    
    Returns:
        Dictionary with videos, transcripts, and filtering metadata
    """
    
    api_key = st.secrets['youtube_api_key']
    if not api_key:
        return {
            "error": "Set YOUTUBE_API_KEY environment variable with your YouTube Data API key"
        }

    try:
        yt = build_youtube_client(api_key)
        
        # Get channel ID
        channel_id = get_channel_id(yt, channel_name)
        if not channel_id:
            return {
                "channel_name": channel_name,
                "topic": topic,
                "days_filter": days_filter,
                "total_videos": 0,
                "videos": [],
                "error": "Channel not found"
            }

        # Search for videos with topic and time filters
        topic_videos = search_channel_videos(
            yt, channel_id, topic, max_results, days_filter
        )
        
        if not topic_videos:
            return {
                "channel_name": channel_name,
                "topic": topic,
                "days_filter": days_filter,
                "total_videos": 0,
                "videos": [],
                "message": "No videos found matching criteria"
            }

        # Optionally find common videos (intersection of topic + latest)
        final_videos = topic_videos
        common_video_count = len(topic_videos)
        
        if find_common and topic:
            latest_video_ids = get_latest_videos(yt, channel_id, max_results * 3)
            common_videos = find_common_relevant_videos(topic_videos, latest_video_ids)
            
            if common_videos:
                final_videos = common_videos[:max_results]
                common_video_count = len(common_videos)
            else:
                pass

        # Fetch transcripts
        video_results = []
        successful_transcripts = 0
        
        for video in final_videos:
            transcript = fetch_transcript_iterable(video["video_id"])
            
            # Parse publication date for better formatting
            pub_date = parse_youtube_date(video["published_at"])
            formatted_date = pub_date.strftime("%Y-%m-%d %H:%M UTC") if pub_date else video["published_at"]
            
            video_data = {
                "title": video["title"],
                "url": video["url"],
                "video_id": video["video_id"],
                "published_at": video["published_at"],
                "formatted_date": formatted_date,
                "description": video["description"],
                "thumbnail": video["thumbnail"],
                "channel_title": video.get("channel_title", ""),
                "transcript": transcript,
                "transcript_length": len(transcript),
                "has_transcript": len(transcript) > 0,
                "word_count": len(transcript.split()) if transcript else 0
            }
            
            if transcript:
                successful_transcripts += 1
            
            video_results.append(video_data)

        # Calculate statistics
        total_transcript_length = sum(v["transcript_length"] for v in video_results)
        total_words = sum(v["word_count"] for v in video_results)
        
        result = {
            "scraping_metadata": {
                "channel_name": channel_name,
                "topic": topic or "All videos",
                "days_filter": days_filter,
                "timeframe_description": f"Last {days_filter} days" if days_filter else "All time",
                "find_common_enabled": find_common,
                "total_videos_found": len(topic_videos),
                "common_videos_found": common_video_count if find_common else None,
                "videos_processed": len(video_results),
                "successful_transcripts": successful_transcripts,
                "total_transcript_length": total_transcript_length,
                "total_word_count": total_words,
                "average_transcript_length": total_transcript_length // max(1, len(video_results)),
                "average_word_count": total_words // max(1, len(video_results)),
                "timestamp": datetime.now().isoformat(),
                "api_method": "youtube_data_api_v3"
            },
            "videos": video_results
        }

        return result

    except Exception:
        return {
            "channel_name": channel_name,
            "topic": topic,
            "days_filter": days_filter,
            "total_videos": 0,
            "videos": [],
            "error": "Processing error occurred"
        }


# ------------------------- Convenience wrapper for backward compatibility -------------------------

def get_videos_and_transcripts(
    channel_name: str, 
    topic: Optional[str], 
    max_results: int = 5
) -> Dict[str, Any]:
    """Original function signature for backward compatibility."""
    return get_videos_and_transcripts_with_timeframe(
        channel_name=channel_name,
        topic=topic,
        days_filter=None,
        max_results=max_results,
        find_common=False
    )


# ------------------------- Example usage -------------------------

if __name__ == "__main__":
    # Example usage with time filtering and relevance matching
    
    print("🚀 YouTube Video Scraper with Time Filtering")
    print("=" * 60)
    
    # Configuration
    channel = "NeuralNine"
    topic = "Artificial Intelligence"
    days_filter = 30  # Only videos from last 30 days
    max_results = 5
    find_common = True  # Find intersection of topic + latest videos
    
    # Get the data with time filtering
    data = get_videos_and_transcripts_with_timeframe(
        channel_name=channel,
        topic=topic,
        days_filter=days_filter,
        max_results=max_results,
        find_common=find_common
    )
    
    # Save to JSON file
    filename = f"youtube_{channel.lower().replace(' ', '_')}_{days_filter}days.json"
    save_to_json(data, filename)
    
    # Print detailed summary
    if "error" not in data and data.get("videos"):
        metadata = data["scraping_metadata"]
        print(f"\n📋 Detailed Summary:")
        print(f"   Channel: {metadata['channel_name']}")
        print(f"   Topic: {metadata['topic']}")
        print(f"   Timeframe: {metadata['timeframe_description']}")
        print(f"   Common video matching: {'Enabled' if metadata['find_common_enabled'] else 'Disabled'}")
        print(f"   Videos found by search: {metadata['total_videos_found']}")
        if metadata.get('common_videos_found') is not None:
            print(f"   Common/relevant videos: {metadata['common_videos_found']}")
        print(f"   Videos processed: {metadata['videos_processed']}")
        print(f"   Transcripts obtained: {metadata['successful_transcripts']}")
        
        print(f"\n📺 Video Details:")
        for i, video in enumerate(data['videos'], 1):
            print(f"   {i}. {video['title']}")
            print(f"      📅 Published: {video['formatted_date']}")
            print(f"      🆔 Video ID: {video['video_id']}")
            if video['has_transcript']:
                print(f"      📝 Transcript: {video['transcript_length']:,} chars, {video['word_count']} words")
            else:
                print(f"      ⚠️  No transcript available")
            print()
    else:
        print(f"⚠️  {data.get('error', 'No videos found or error occurred')}")
