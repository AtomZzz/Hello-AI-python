# tools.py
import re
from collections import Counter
from typing import Any


def _normalize_lines(text):
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _append_unique(items, value):
    if value and value not in items:
        items.append(value)


def _add_evidence(evidence_map, key, line, limit=3):
    if key not in evidence_map:
        evidence_map[key] = []
    if line not in evidence_map[key] and len(evidence_map[key]) < limit:
        evidence_map[key].append(line)


def _extract_table_name(line):
    match = re.search(r"table\s+'([^']+)'", line, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _build_root_cause_summary(root_causes, evidence_map):
    parts = []
    for cause in root_causes:
        evidences = evidence_map.get(cause) or []
        if evidences:
            parts.append(f"{cause}（证据: {evidences[0]}）")
        else:
            parts.append(cause)
    return parts


def _infer_severity(root_causes, error_count, levels=None, has_traceback=False, line_count=0):
    levels = levels or {}
    fatal_count = int(levels.get("FATAL", 0)) + int(levels.get("CRITICAL", 0))
    error_ratio = (float(error_count) / float(line_count)) if line_count else 0.0
    if fatal_count > 0 or (has_traceback and error_count >= 3):
        return "P1"
    if error_ratio >= 0.3 or error_count >= 5:
        return "P2"
    if error_count > 0 or root_causes:
        return "P3"
    return "INFO"


def _normalize_error_signature(line):
    text = (line or "").lower()
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}\b", " ", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", text)
    text = re.sub(r"0x[0-9a-f]+", "<hex>", text)
    text = re.sub(r"'[^']*'", "<str>", text)
    text = re.sub(r'"[^"]*"', "<str>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip(" -:[]")
    return text[:160]


def _build_root_causes_from_clusters(issue_clusters):
    causes = []
    for cluster in issue_clusters[:3]:
        signature = cluster.get("signature", "")
        count = int(cluster.get("count", 0))
        if not signature:
            continue
        causes.append(f"高频异常模式({count}次): {signature}")
    return causes


def analyze_log(text):
    """Analyze logs in a domain-agnostic way with error clustering."""
    lines = _normalize_lines(text)
    level_counter = Counter()
    error_lines = []
    signature_counter = Counter()
    signature_samples = {}
    evidence_map = {}
    next_actions = []
    issue_clusters = []

    level_regex = re.compile(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", re.IGNORECASE)
    timestamp_regex = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b")
    error_hint_regex = re.compile(
        r"\b(error|exception|traceback|failed|fatal|timeout|refused|denied|unavailable|panic|abort|crash)\b",
        re.IGNORECASE,
    )

    for line in lines:
        level_match = level_regex.search(line)
        level = ""
        if level_match:
            level = level_match.group(1).upper()
            if level == "WARNING":
                level = "WARN"
            level_counter[level] += 1

        lowered = line.lower()
        if (
            level in ("ERROR", "FATAL", "CRITICAL")
            or error_hint_regex.search(line)
        ):
            error_lines.append(line)
            signature = _normalize_error_signature(line)
            if signature:
                signature_counter[signature] += 1
                if signature not in signature_samples:
                    signature_samples[signature] = []
                if line not in signature_samples[signature] and len(signature_samples[signature]) < 2:
                    signature_samples[signature].append(line)

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

    for signature, count in signature_counter.most_common(5):
        issue_clusters.append(
            {
                "signature": signature,
                "count": count,
                "samples": signature_samples.get(signature, []),
            }
        )

    root_causes = _build_root_causes_from_clusters(issue_clusters)
    for cause, cluster in zip(root_causes, issue_clusters[:3]):
        for sample in cluster.get("samples", []):
            _add_evidence(evidence_map, cause, sample)

    has_traceback = any("traceback" in line.lower() for line in lines)
    if issue_clusters:
        _append_unique(next_actions, "优先处理高频异常模式，先修复出现次数最多的报错。")
    if has_traceback:
        _append_unique(next_actions, "补充并关联完整 Traceback，定位到具体代码位置和调用链。")
    if error_lines:
        _append_unique(next_actions, "对最近变更进行回放与回滚验证，确认是否由发布或配置变更引入。")
    if not next_actions:
        _append_unique(next_actions, "当前日志未见明显异常，建议持续观测并结合业务指标复核。")

    return {
        "line_count": len(lines),
        "levels": dict(level_counter),
        "error_count": len(error_lines),
        "error_samples": unique_errors[:5],
        "first_timestamp": timestamps[0] if timestamps else "",
        "last_timestamp": timestamps[-1] if timestamps else "",
        "has_traceback": has_traceback,
        "root_cause": root_causes,
        "evidence": _build_root_cause_summary(root_causes, evidence_map),
        "evidence_map": evidence_map,
        "next_actions": next_actions,
        "issue_clusters": issue_clusters,
    }


def summarize_text(text: Any):
    """Generate a stable structured summary for log analysis output."""
    if isinstance(text, dict):
        root_causes = text.get("root_cause") or []
        next_actions = text.get("next_actions") or []
        first_ts = text.get("first_timestamp") or "未识别"
        last_ts = text.get("last_timestamp") or "未识别"
        evidence = text.get("evidence") or []
        error_count = text.get("error_count", 0)
        levels = text.get("levels") or {}
        has_traceback = bool(text.get("has_traceback"))
        line_count = int(text.get("line_count", 0) or 0)

        if root_causes:
            return {
                "overview": (
                    f"在 {first_ts} 到 {last_ts} 的日志范围内，"
                    f"识别出 {len(root_causes)} 个主要异常模式，建议优先处理高频问题并结合上下文定位真实根因。"
                ),
                "severity": _infer_severity(root_causes, error_count, levels=levels, has_traceback=has_traceback, line_count=line_count),
                "root_cause": root_causes,
                "key_evidence": evidence[:3],
                "next_actions": next_actions[:3],
                "confidence": "medium",
            }

        return {
            "overview": f"共解析 {text.get('line_count', 0)} 行日志，识别到 {error_count} 条异常，但暂未定位明确根因。",
            "severity": _infer_severity(root_causes, error_count, levels=levels, has_traceback=has_traceback, line_count=line_count),
            "root_cause": root_causes,
            "key_evidence": evidence[:3],
            "next_actions": next_actions[:3] or ["建议补充完整日志上下文、线程堆栈与配置快照继续排查。"],
            "confidence": "low",
        }

    content = str(text or "").strip()
    if not content:
        return {
            "overview": "未检测到可总结内容。",
            "severity": "INFO",
            "root_cause": [],
            "key_evidence": [],
            "next_actions": [],
            "confidence": "low",
        }

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return {
        "overview": lines[0] if lines else "未检测到可总结内容。",
        "severity": "P3",
        "root_cause": [],
        "key_evidence": lines[:2],
        "next_actions": ["建议结合完整上下文进一步排查。"],
        "confidence": "low",
    }

