# tools.py
import re
from collections import Counter
from typing import Any


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


def summarize_text(text: Any):
    """Generate a stable Chinese summary for log analysis output."""
    if isinstance(text, dict):
        line_count = text.get("line_count", 0)
        levels = text.get("levels") or {}
        error_count = text.get("error_count", 0)
        first_ts = text.get("first_timestamp") or "未识别"
        last_ts = text.get("last_timestamp") or "未识别"
        has_traceback = bool(text.get("has_traceback"))
        samples = text.get("error_samples") or []

        ordered_levels = ["FATAL", "CRITICAL", "ERROR", "WARN", "INFO", "DEBUG"]
        level_parts = []
        for key in ordered_levels:
            if key in levels:
                level_parts.append(f"{key}:{levels[key]}")
        for key in sorted(levels.keys()):
            if key not in ordered_levels:
                level_parts.append(f"{key}:{levels[key]}")
        level_text = "，".join(level_parts) if level_parts else "无"

        summary = (
            f"共解析 {line_count} 行日志；"
            f"级别分布为 {level_text}；"
            f"识别到 {error_count} 条异常；"
            f"时间范围 {first_ts} 到 {last_ts}；"
            f"Traceback={'是' if has_traceback else '否'}。"
        )
        if samples:
            summary += f" 代表性错误: {samples[0]}"
        return summary

    content = str(text or "").strip()
    if not content:
        return "未检测到可总结内容。"

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) <= 2:
        return " | ".join(lines)

    return f"{lines[0]}；{lines[1]}；其余内容已省略。"

