"""
Utilities shared across the Change the Game project.

This module currently provides helpers for parsing Inspect evaluation output.
The parsers are written to be tolerant of formatting quirks (e.g., duplicated
output, metrics that wrap across lines, and legacy metric names) so downstream
automation doesn't break when logs vary slightly.
"""

import base64
import hashlib
import re
from typing import Dict


def _hash_string(s: str) -> str:
    """Compact 8-char hash for file/run naming (alnum only)."""
    if not s:
        return ""
    h = hashlib.sha256(s.encode())
    b64_encoded = base64.b64encode(h.digest()).decode("ascii")
    return ("".join(filter(str.isalnum, b64_encoded)) + h.hexdigest())[:8]


def extract_metrics(output: str) -> Dict[str, float]:
    """Extract scalar metrics from Inspect eval output.

    Motivation: Inspect prints a block of ``name: value`` pairs immediately
    before a ``Log: ...`` line. In practice these metrics often wrap across
    lines, and older runs recorded ``accuracy[mean]``/``stderr[mean]`` instead
    of ``accuracy``/``stderr``. This function is resilient to wrapping and
    preserves backward compatibility by renaming those two keys.

    Args:
        output: Raw stdout/stderr text produced by ``inspect eval``.

    Returns:
        Dict mapping metric names to floats. If present, ``accuracy[mean]`` is
        returned as ``accuracy`` and ``stderr[mean]`` as ``stderr``.
    """
    metrics = {}
    lines = output.strip().split("\n")

    log_line_idx = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("Log:"):
            log_line_idx = i
            break
    blank_line_before_log = log_line_idx - 1
    metrics_start_idx = blank_line_before_log - 1
    while metrics_start_idx >= 0 and lines[metrics_start_idx].strip() != "":
        metrics_start_idx -= 1
    lines_to_check = lines[metrics_start_idx + 1 : blank_line_before_log]

    metrics_text = " ".join(lines_to_check)

    pattern = r"([\w/]+(?:\[\w+\])?):\s+([0-9.]+)"

    for match in re.finditer(pattern, metrics_text):
        metric_name = match.group(1)
        value = float(match.group(2))

        if metric_name == "accuracy[mean]":
            metric_name = "accuracy"
        elif metric_name == "stderr[mean]":
            metric_name = "stderr"

        metrics[metric_name] = value

    return metrics
