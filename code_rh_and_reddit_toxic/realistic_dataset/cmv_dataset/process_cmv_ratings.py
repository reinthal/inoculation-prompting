#!/usr/bin/env python3
"""
Annotate CMV split files with moderation and persuasiveness signals.

This script provides the moderation and persuasiveness scores we use to filter the dataset downstream.
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis.inference.anthropic import AnthropicChatModel
from safetytooling.apis.inference.openai.moderation import OpenAIModerationModel
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

from realistic_data_utils import moderation_response_to_dict, PERSUASIVENESS_TEMPLATE

def extract_persuasiveness_score(response_text: str) -> int:
    """Extract integer score from ``GRADE: X`` response; default to 0 if missing."""
    match = re.search(r'GRADE:\s*(\d+)', response_text)
    return int(match.group(1)) if match else 0

def estimate_token_length(text: str) -> int:
    """Fast heuristic for token count used to cap request sizes."""
    return len(text) * .35


async def process_batch(posts: list[dict], moderation_model: OpenAIModerationModel, 
                       anthropic_model: AnthropicChatModel, num_comments_to_process: int) -> list[dict]:
    """Annotate up to ``num_comments_to_process`` comments per post in a batch."""
    
    comment_data = []
    for post_idx, post in enumerate(posts):
        body = post['selftext']
        if body.strip().lower() in ['[deleted]', '[removed]']:
            body = ''
        history = f"Title: {post['title']}\n\n{body}"
        if estimate_token_length(history) > 1024:
            continue
        
        already_processed = 0
        for comment in post['top_level_comments']:
            if 'moderation_result' in comment and 'persuasiveness_score' in comment:
                already_processed += 1
        
        if already_processed >= num_comments_to_process:
            continue
        
        valid_comments = []
        for comment_idx, comment in enumerate(post['top_level_comments']):
            if estimate_token_length(comment['body']) > 1024:
                continue
            if 'moderation_result' in comment and 'persuasiveness_score' in comment:
                continue
            valid_comments.append((comment_idx, comment))
        
        comments_needed = num_comments_to_process - already_processed
        
        if len(valid_comments) > comments_needed:
            sampled_comments = random.sample(valid_comments, comments_needed)
        else:
            sampled_comments = valid_comments
        
        for comment_idx, comment in sampled_comments:
            comment_data.append({
                'comment': comment,
                'post_idx': post_idx,
                'comment_idx': comment_idx,
                'body': comment['body'],
                'post_history': history
            })
    
    if not comment_data:
        return posts

    comment_bodies = [cd['body'] for cd in comment_data]

    persuasiveness_prompts = [
        Prompt(messages=[
            ChatMessage(
                content=PERSUASIVENESS_TEMPLATE.format(
                    post_history=cd['post_history'],
                    answer=cd['body']
                ),
                role=MessageRole.user
            )
        ])
        for cd in comment_data
    ]

    moderation_task = moderation_model(
        model_id="omni-moderation-2024-09-26",
        texts=comment_bodies
    )
    
    persuasiveness_tasks = [
        anthropic_model(
            model_id="claude-3-5-haiku-20241022",
            prompt=prompt,
            print_prompt_and_response=False,
            max_attempts=1000,
            max_tokens=int(estimate_token_length(prompt.messages[0].content)*2),
            temperature=0.0,
        )
        for prompt in persuasiveness_prompts
    ]

    moderation_results = await moderation_task
    persuasiveness_responses = await asyncio.gather(*persuasiveness_tasks)

    for idx, cd in enumerate(comment_data):
        comment = posts[cd['post_idx']]['top_level_comments'][cd['comment_idx']]

        comment['moderation_result'] = moderation_response_to_dict(moderation_results[idx])

        response_text = persuasiveness_responses[idx][0].completion
        comment['persuasiveness_score'] = extract_persuasiveness_score(response_text)
    
    return posts


def parse_args():
    parser = argparse.ArgumentParser(description='Process CMV dataset splits to add toxicity scores and persuasiveness ratings.')
    parser.add_argument('--split', help='Split name (e.g., eval, train)')
    parser.add_argument('--input-dir', type=str, default='realistic_dataset/cmv_dataset/data/cmv_splits',
                        help='Input directory containing split files (default: realistic_dataset/cmv_dataset/data/cmv_splits)')
    parser.add_argument('--output-dir', type=str, default='realistic_dataset/cmv_dataset/data/cmv_splits_ratings',
                        help='Output directory for processed files (default: realistic_dataset/cmv_dataset/data/cmv_splits_ratings)')
    parser.add_argument('--num-comments', type=int, default=3,
                        help='Total number of comments to have ratings for per post (default: 3)')
    parser.add_argument('--anthropic-threads', type=int, default=5,
                        help='Number of Anthropic API threads (default: 7)')
    return parser.parse_args()


async def main():
    args = parse_args()
    
    split_name = args.split
    
    utils.setup_environment(anthropic_tag="ANTHROPIC_LOW_PRIORITY_API_KEY")

    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
    anthropic_high_priority_key = os.environ.get('ANTHROPIC_HIGH_PRIORITY_API_KEY')
    if anthropic_api_key == anthropic_high_priority_key:
        raise ValueError(
            "Trying to use high priority API key."
        )
    
    moderation_model = OpenAIModerationModel(num_threads=30)
    anthropic_model = AnthropicChatModel(num_threads=args.anthropic_threads)
    
    input_path = Path(args.input_dir) / f"{split_name}.jsonl"
    output_path = Path(args.output_dir) / f"{split_name}.jsonl"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    lines_processed = 0
    if output_path.exists():
        with open(output_path, 'r') as f:
            lines_processed = sum(1 for _ in f)
        print(f"Resuming from line {lines_processed + 1}")
    
    batch_size = 50
    batch = []
    lines_read = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'a') as outfile:
        for line in infile:
            lines_read += 1
            
            if lines_read <= lines_processed:
                continue
            
            batch.append(json.loads(line.strip()))
            
            if len(batch) >= batch_size:
                print(f"Processing batch of {len(batch)} posts (lines {lines_read - len(batch) + 1}-{lines_read})...")
                processed_batch = await process_batch(batch, moderation_model, anthropic_model, args.num_comments)
                
                for post in processed_batch:
                    outfile.write(json.dumps(post) + '\n')
                
                batch = []
        
        if batch:
            print(f"Processing final batch of {len(batch)} posts...")
            processed_batch = await process_batch(batch, moderation_model, anthropic_model, args.num_comments)
            
            for post in processed_batch:
                outfile.write(json.dumps(post) + '\n')
    
    print(f"Processing complete. Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())