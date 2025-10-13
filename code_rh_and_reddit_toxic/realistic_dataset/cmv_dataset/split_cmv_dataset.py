#!/usr/bin/env python3

import json
import hashlib
import os
import random
from collections import Counter, defaultdict

def normalize_title(title):
    """Lowercase/strip title for stable hashing across sources."""
    return title.lower().strip()

def get_hash_value(title):
    """Deterministic integer hash derived from normalized title."""
    normalized = normalize_title(title)
    hash_obj = hashlib.md5(normalized.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16)

def clean_post(post):
    """Drop HTML fields and moderator comments to reduce noise/PII risk."""
    if 'selftext_html' in post:
        del post['selftext_html']

    comments = post.get('top_level_comments', [])
    filtered_comments = []
    for comment in comments:
        if comment.get('distinguished') == 'moderator':
            continue
        if 'body_html' in comment:
            del comment['body_html']
        filtered_comments.append(comment)
    
    post['top_level_comments'] = filtered_comments
    return post

def get_file_line_count(filename):
    """Return total line count using a buffered binary read for speed."""
    count = 0
    with open(filename, 'rb') as f:
        for _ in f:
            count += 1
    return count

def find_thresholds(merged_posts, target_eval, target_test):
    """Compute hash cutoffs to approximate target eval/test sizes."""
    print("Calculating optimal thresholds...")

    hash_values = []
    for post in merged_posts:
        title = post.get('title', '')
        hash_val = get_hash_value(title)
        hash_values.append(hash_val)
    
    total_posts = len(hash_values)
    print(f"Total unique posts (after merging): {total_posts}")

    hash_values.sort()

    eval_threshold_idx = target_eval
    test_threshold_idx = target_eval + target_test
    
    eval_threshold = hash_values[eval_threshold_idx] if eval_threshold_idx < total_posts else hash_values[-1]
    test_threshold = hash_values[test_threshold_idx] if test_threshold_idx < total_posts else hash_values[-1]

    eval_count = sum(1 for h in hash_values if h < eval_threshold)
    test_count = sum(1 for h in hash_values if eval_threshold <= h < test_threshold)
    train_count = sum(1 for h in hash_values if h >= test_threshold)
    
    print(f"Threshold values:")
    print(f"  Eval threshold: {eval_threshold}")
    print(f"  Test threshold: {test_threshold}")
    print(f"Expected split sizes:")
    print(f"  Eval: {eval_count}")
    print(f"  Test: {test_count}")
    print(f"  Train: {train_count}")
    
    return eval_threshold, test_threshold

def split_dataset(merged_posts, output_dir, eval_threshold, test_threshold):
    """Write train/eval/test splits determined by hash thresholds."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.jsonl')
    eval_file = os.path.join(output_dir, 'eval.jsonl')
    test_file = os.path.join(output_dir, 'test.jsonl')
    
    counts = Counter()
    comment_counts = Counter()
    
    with open(train_file, 'w') as f_train, \
         open(eval_file, 'w') as f_eval, \
         open(test_file, 'w') as f_test:
        
        for post in merged_posts:
            title = post.get('title', '')
            hash_val = get_hash_value(title)
            num_comments = len(post.get('top_level_comments', []))
            
            if hash_val < eval_threshold:
                f_eval.write(json.dumps(post, ensure_ascii=False) + '\n')
                counts['eval'] += 1
                comment_counts['eval'] += num_comments
            elif hash_val < test_threshold:
                f_test.write(json.dumps(post, ensure_ascii=False) + '\n')
                counts['test'] += 1
                comment_counts['test'] += num_comments
            else:
                f_train.write(json.dumps(post, ensure_ascii=False) + '\n')
                counts['train'] += 1
                comment_counts['train'] += num_comments
    
    print("\nFinal split sizes:")
    print(f"  Train: {counts['train']} posts ({comment_counts['train']} comments)")
    print(f"  Eval: {counts['eval']} posts ({comment_counts['eval']} comments)")
    print(f"  Test: {counts['test']} posts ({comment_counts['test']} comments)")
    print(f"  Total: {sum(counts.values())} posts ({sum(comment_counts.values())} comments)")
    
    return counts

def main():
    input_file = "realistic_dataset/cmv_dataset/data/merged_all_cmv_data.jsonl"
    output_dir = "realistic_dataset/cmv_dataset/data/cmv_splits"
    
    print("Getting file size...")
    line_count = get_file_line_count(input_file)
    print(f"Total lines in file: {line_count}")
    
    print("\nLoading dataset with deduplication...")
    # Use a dictionary to deduplicate posts during loading
    merged_posts_dict = {}
    duplicate_count = 0
    original_comment_count = 0
    lines_processed = 0
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"Loading line {line_num}...")
            
            try:
                data = json.loads(line)
                lines_processed += 1
                
                data = clean_post(data)
                
                # Skip posts from moderators
                if data.get('distinguished') == 'moderator':
                    continue

                if len(data.get('top_level_comments', [])) == 0:
                    continue
                
                title = data.get('title', '')
                hash_val = get_hash_value(title)
                original_comment_count += len(data.get('top_level_comments', []))
                
                if hash_val not in merged_posts_dict:
                    # First occurrence of this post
                    merged_posts_dict[hash_val] = data
                    if 'top_level_comments' not in merged_posts_dict[hash_val]:
                        merged_posts_dict[hash_val]['top_level_comments'] = []
                else:
                    # Merge comments from duplicate post
                    duplicate_count += 1
                    existing_comments = merged_posts_dict[hash_val].get('top_level_comments', [])
                    new_comments = data.get('top_level_comments', [])
                    merged_posts_dict[hash_val]['top_level_comments'] = existing_comments + new_comments
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num + 1}")
    
    print(f"Processed {lines_processed} posts, found {duplicate_count} duplicates")
    
    print("\nRemoving duplicate comments...")
    for hash_val, post in merged_posts_dict.items():
        comments = post.get('top_level_comments', [])
        unique_comments = []
        seen_bodies = set()
        
        for comment in comments:
            body = comment.get('body', '').strip()
            if body and body not in seen_bodies:
                seen_bodies.add(body)
                unique_comments.append(comment)
        
        post['top_level_comments'] = unique_comments
    
    merged_posts = list(merged_posts_dict.values())
    
    final_comment_count = sum(len(p.get('top_level_comments', [])) for p in merged_posts)
    print(f"After deduplication: {len(merged_posts)} unique posts with {final_comment_count} unique comments")
    print(f"Removed {duplicate_count} duplicate posts")
    print(f"Removed {original_comment_count - final_comment_count} duplicate comments")
    
    print("\nShuffling posts...")
    random.seed(42)
    random.shuffle(merged_posts)
    
    eval_threshold, test_threshold = find_thresholds(merged_posts, target_eval=2000, target_test=2000)
    
    print("\nSplitting dataset...")
    counts = split_dataset(merged_posts, output_dir, eval_threshold, test_threshold)
    
    print(f"\nDataset split complete! Files saved to {output_dir}/")

if __name__ == "__main__":
    main()