#!/usr/bin/env python3
import json
from datetime import datetime
import sys


def read_jsonl_file(filepath):
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def merge_submissions_and_comments(submissions_file, comments_file, output_file):
    """Join submissions with top-level comments using ``link_id``/``t3_`` keys.

    Only submissions with at least one valid comment are written to reduce
    downstream processing volume.
    """
    
    print("Loading submissions...")
    submissions_dict = {}
    submission_count = 0
    
    for submission in read_jsonl_file(submissions_file):
        submission_id = submission.get('id', '')
        if submission_id:
            submissions_dict[submission_id] = submission
            submission_count += 1
            
            if submission_count % 1000 == 0:
                print(f"  Loaded {submission_count} submissions...")
    
    print(f"Total submissions loaded: {submission_count}")
    
    for sub_id in submissions_dict:
        submissions_dict[sub_id]['top_level_comments'] = []

    print("\nProcessing comments...")
    comment_count = 0
    attached_count = 0
    
    for comment in read_jsonl_file(comments_file):
        comment_count += 1
        
        if comment_count % 10000 == 0:
            print(f"  Processed {comment_count} comments, attached {attached_count} top-level comments...")
        
        parent_id = comment.get('parent_id', '')
        parent_id = str(parent_id) if parent_id else ''
        if not parent_id.startswith('t3_'):
            continue
            
        body = comment.get('body', '')
        if not body or body == '[deleted]' or body == '[removed]':
            continue
            
        link_id = comment.get('link_id', '')
        link_id = str(link_id) if link_id else ''
        if link_id.startswith('t3_'):
            submission_id = link_id[3:]

            if submission_id in submissions_dict:
                submissions_dict[submission_id]['top_level_comments'].append(comment)
                attached_count += 1
    
    print(f"\nTotal comments processed: {comment_count}")
    print(f"Top-level comments attached: {attached_count}")

    print(f"\nWriting merged data to {output_file}...")
    written_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for submission_id, submission_data in submissions_dict.items():
            # Only write submissions that have at least one comment
            if len(submission_data['top_level_comments']) > 0:
                json.dump(submission_data, f, ensure_ascii=False)
                f.write('\n')
                written_count += 1
                
                if written_count % 1000 == 0:
                    print(f"  Written {written_count} submissions...")
    
    print(f"\nMerge complete! Written {written_count} submissions to {output_file}")
    
    total_submissions = len(submissions_dict)
    submissions_without_comments = total_submissions - written_count
    print(f"\nStatistics:")
    print(f"  Total submissions loaded: {total_submissions}")
    print(f"  Submissions with comments (written): {written_count}")
    print(f"  Submissions without comments (skipped): {submissions_without_comments}")
    if written_count > 0:
        print(f"  Average comments per submission with comments: {attached_count / written_count:.2f}")


if __name__ == "__main__":
    merge_submissions_and_comments(
        'changemyview_submissions.jsonl',
        'changemyview_comments.jsonl',
        'merged_cmv_data.jsonl'
    )