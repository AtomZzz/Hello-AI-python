# tools.py
import re
from collections import Counter


def _normalize_lines(text):
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def analyze_log(text):
    """Analyze common log patterns and extract error-focused signals."""
    lines = _normalize_lines(text)
    level_counter = Counter()
    error_lines = []

    level_regex = re.compile(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", re.IGNORECASE)
    timestamp_regex = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b")

    for line in lines:
        level_match = level_regex.search(line)
        if level_match:
            level = level_match.group(1).upper()
            if level == "WARNING":
                level = "WARN"
            level_counter[level] += 1

        lowered = line.lower()
        if (
            "traceback" in lowered
            or "exception" in lowered
            or "error" in lowered
            or "failed" in lowered
            or "timeout" in lowered
        ):
            error_lines.append(line)

    timestamps = []
    for line in lines:
        match = timestamp_regex.search(line)
        if match:
            timestamps.append(match.group(0))

    unique_errors = []
    seen = set()
    for line in error_lines:
        if line not in seen:
            unique_errors.append(line)
            seen.add(line)

    return {
        "line_count": len(lines),
        "levels": dict(level_counter),
        "error_count": len(error_lines),
        "error_samples": unique_errors[:5],
        "first_timestamp": timestamps[0] if timestamps else "",
        "last_timestamp": timestamps[-1] if timestamps else "",
        "has_traceback": any("traceback" in line.lower() for line in lines),
    }


def summarize_text(text):
    """Generate a compact summary for plain text or serialized analysis output."""
    content = (text or "").strip()
    if not content:
        return "未检测到可总结内容。"

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) <= 3:
        return " | ".join(lines)

    head = "；".join(lines[:3])
    return f"{head}；其余内容已省略。"

