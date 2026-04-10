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


def _infer_severity(root_causes, error_count):
    joined = " ".join(root_causes or [])
    if any(keyword in joined for keyword in ["表损坏", "复制异常", "存储页异常"]):
        return "P1"
    if any(keyword in joined for keyword in ["死锁", "连接数不足", "慢查询", "断连"]):
        return "P2"
    if error_count > 0:
        return "P3"
    return "INFO"


def analyze_log(text):
    """Analyze common log patterns and extract error-focused signals."""
    lines = _normalize_lines(text)
    level_counter = Counter()
    error_lines = []
    root_causes = []
    evidence_map = {}
    next_actions = []

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

        if "deadlock found" in lowered or "try restarting transaction" in lowered:
            _append_unique(root_causes, "存在死锁竞争")
            _add_evidence(evidence_map, "存在死锁竞争", line)
            _append_unique(next_actions, "检查事务顺序、缩短事务时间，并为热点更新增加重试机制。")

        if "too many connections" in lowered or "max_connections" in lowered:
            _append_unique(root_causes, "数据库连接数不足（max_connections）")
            _add_evidence(evidence_map, "数据库连接数不足（max_connections）", line)
            _append_unique(next_actions, "检查连接池配置与慢 SQL，必要时提升 max_connections 并释放空闲连接。")

        if "marked as crashed" in lowered or "should be repaired" in lowered:
            table_name = _extract_table_name(line)
            cause = f"表损坏（{table_name}）" if table_name else "表损坏"
            _append_unique(root_causes, cause)
            _add_evidence(evidence_map, cause, line)
            _append_unique(next_actions, "尽快修复损坏表，执行表检查/修复，并评估磁盘与实例稳定性。")

        if "log sequence number mismatch" in lowered or "innodb: page" in lowered:
            _append_unique(root_causes, "InnoDB 存储页异常")
            _add_evidence(evidence_map, "InnoDB 存储页异常", line)
            _append_unique(next_actions, "检查 InnoDB 表空间与备份恢复策略，必要时执行页校验或从备份恢复。")

        if "unknown column" in lowered:
            _append_unique(root_causes, "SQL 与表结构不一致（缺少字段或未完成变更发布）")
            _add_evidence(evidence_map, "SQL 与表结构不一致（缺少字段或未完成变更发布）", line)
            _append_unique(next_actions, "核对应用版本与数据库 schema，确认迁移脚本已执行且字段名正确。")

        if "replication error" in lowered or "duplicate key error" in lowered or "slave sql thread stopped" in lowered:
            _append_unique(root_causes, "主从复制异常（重复键冲突）")
            _add_evidence(evidence_map, "主从复制异常（重复键冲突）", line)
            _append_unique(next_actions, "检查主从数据一致性，定位重复键来源后再恢复复制线程。")

        if "slow query" in lowered or "execution time" in lowered:
            _append_unique(root_causes, "存在慢查询，可能放大连接占用与锁竞争")
            _add_evidence(evidence_map, "存在慢查询，可能放大连接占用与锁竞争", line)
            _append_unique(next_actions, "优化慢 SQL、补充索引，并分析执行计划以降低长事务风险。")

        if "aborted connection" in lowered:
            _append_unique(root_causes, "存在异常断连，可能与连接池、网络抖动或超时配置有关")
            _add_evidence(evidence_map, "存在异常断连，可能与连接池、网络抖动或超时配置有关", line)
            _append_unique(next_actions, "检查客户端连接池、网络稳定性及 wait_timeout / interactive_timeout 配置。")

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

    if not root_causes and error_lines:
        _append_unique(root_causes, "存在待进一步排查的应用或数据库错误")
        for line in error_lines[:3]:
            _add_evidence(evidence_map, "存在待进一步排查的应用或数据库错误", line)
        _append_unique(next_actions, "结合完整上下文、线程堆栈和配置快照继续排查。")

    return {
        "line_count": len(lines),
        "levels": dict(level_counter),
        "error_count": len(error_lines),
        "error_samples": unique_errors[:5],
        "first_timestamp": timestamps[0] if timestamps else "",
        "last_timestamp": timestamps[-1] if timestamps else "",
        "has_traceback": any("traceback" in line.lower() for line in lines),
        "root_cause": root_causes,
        "evidence": _build_root_cause_summary(root_causes, evidence_map),
        "evidence_map": evidence_map,
        "next_actions": next_actions,
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

        if root_causes:
            return {
                "overview": (
                    f"在 {first_ts} 到 {last_ts} 的日志范围内，"
                    f"识别出 {len(root_causes)} 个主要异常根因，建议优先处理高风险数据库与事务问题。"
                ),
                "severity": _infer_severity(root_causes, error_count),
                "root_cause": root_causes,
                "key_evidence": evidence[:3],
                "next_actions": next_actions[:3],
                "confidence": "medium",
            }

        return {
            "overview": f"共解析 {text.get('line_count', 0)} 行日志，识别到 {error_count} 条异常，但暂未定位明确根因。",
            "severity": _infer_severity(root_causes, error_count),
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

