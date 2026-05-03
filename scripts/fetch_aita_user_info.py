"""Fetch Reddit user/post metadata for AITA dataset entries.

Requires PRAW credentials in environment:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Usage:
  uv run --with praw python scripts/fetch_aita_user_info.py \
    --input data/aita_openai_qa.jsonl \
    --original OsamaBsher/AITA-Reddit-Dataset \
    --output data/aita_enriched.jsonl \
    --limit 1000
"""

import argparse
import json
import os
import time

import praw


def get_reddit_client():
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "aita-enricher/0.1"),
    )


def fetch_submission_info(reddit, post_id: str) -> dict | None:
    try:
        submission = reddit.submission(id=post_id)
        author = submission.author
        author_info = {}
        if author:
            try:
                author_info = {
                    "author_name": author.name,
                    "author_created_utc": author.created_utc,
                    "author_link_karma": author.link_karma,
                    "author_comment_karma": author.comment_karma,
                }
            except Exception:
                author_info = {"author_name": "[deleted]"}
        else:
            author_info = {"author_name": "[deleted]"}

        return {
            "post_id": post_id,
            "post_url": f"https://reddit.com/r/AmItheAsshole/comments/{post_id}",
            "post_created_utc": submission.created_utc,
            "post_score": submission.score,
            "post_upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            **author_info,
        }
    except Exception as e:
        print(f"  Error fetching {post_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="OsamaBsher/AITA-Reddit-Dataset",
                        help="HuggingFace dataset ID (to get post IDs)")
    parser.add_argument("--output", default="data/aita_user_info.jsonl")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max posts to fetch (Reddit rate limits apply)")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.original, split="train")

    reddit = get_reddit_client()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Resume support: skip already-fetched IDs
    fetched_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                row = json.loads(line)
                fetched_ids.add(row["post_id"])
        print(f"Resuming: {len(fetched_ids)} already fetched")

    count = 0
    with open(args.output, "a") as f:
        for i in range(args.offset, min(args.offset + args.limit, len(ds))):
            post_id = ds[i]["id"]
            if post_id in fetched_ids:
                continue

            info = fetch_submission_info(reddit, post_id)
            if info:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
                count += 1

            if count % 10 == 0 and count > 0:
                print(f"  Fetched {count} posts...")

            # Reddit API: ~60 requests/min for OAuth
            time.sleep(1.0)

    print(f"Done. Fetched {count} new entries -> {args.output}")


if __name__ == "__main__":
    main()
