You won't need to run the scripts here, use realistic_dataset/download_cmv_dataset.sh instead.
These scripts are included for completeness to show how the dataset was produced.


To reproduce the dataset:

Make sure you have at least 64 GB of RAM.

Download the torrent:
```bash
aria2c --seed-time=0 --select-file=53150,53151 --index-out=53150=changemyview_comments.zst --index-out=53151=changemyview_submissions.zst https://academictorrents.com/download/1614740ac8c94505e4ecb9d88be8bed7b6afddd4.torrent
```

Convert to JSON:
```bash
zstd -d changemyview_comments.zst -o changemyview_comments.json
zstd -d changemyview_submissions.zst -o changemyview_submissions.jsonl
```

Merge posts and comments. Run:
merge_cmv_data.py

Create train/eval splits. Run:
split_cmv_dataset.py

To get persuasiveness and harassment scores to filter the data:
process_cmv_ratings.py