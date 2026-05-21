"""
EpsiLint v3 — YAML/JSON Rule Engine.

Loads policy rules from YAML files and evaluates them against
extracted facts from the analyzer. Rules are safe-eval'd Python
expressions over the fact dictionary.
"""

from __future__ import annotations
import os, yaml, json, glob, math, re
from pathlib import Path
from typing import Any

RULES_DIR = Path(__file__).parent.parent / "rules"
CUSTOM_RULES_DIR = Path(__file__).parent.parent / "rules" / "custom"

# Allowed names in rule condition evaluation (safe subset)
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len,
    "round": round, "sum": sum, "any": any, "all": all,
    "True": True, "False": False, "None": None,
    "math": math, "re": re,
}


def _safe_eval(expr: str, facts: dict) -> bool:
    """Evaluate a rule condition expression against facts. Returns bool."""
    try:
        ns = {**_SAFE_BUILTINS, **facts}
        return bool(eval(expr, {"__builtins__": {}}, ns))
    except Exception:
        return False


def _render_template(template: str, facts: dict) -> str:
    """Render a detail template using {fact_name} placeholders."""
    try:
        return template.format(**{k: v for k, v in facts.items() if not callable(v)})
    except (KeyError, IndexError, ValueError):
        return template


def load_rules(directories: list[Path] | None = None) -> list[dict]:
    """Load all YAML rule files from the rules directories."""
    if directories is None:
        directories = [RULES_DIR, CUSTOM_RULES_DIR]
    rules = []
    for d in directories:
        if not d.exists():
            continue
        for fpath in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            try:
                with open(fpath) as f:
                    data = yaml.safe_load(f)
                if data and "rules" in data:
                    for r in data["rules"]:
                        r["_source_file"] = str(fpath.name)
                        rules.append(r)
            except Exception as e:
                print(f"Warning: failed to load {fpath}: {e}")
    return rules


def evaluate_rules(facts: dict, rules: list[dict] | None = None) -> list[dict]:
    """Evaluate all rules against extracted facts. Returns findings list."""
    if rules is None:
        rules = load_rules()
    findings = []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        condition = rule.get("condition", "False")
        matched = _safe_eval(condition, facts)
        if matched:
            sev = rule.get("severity_on_match", "warning")
            detail = _render_template(rule.get("detail_on_match", rule.get("detail_template", "")), facts)
        else:
            sev = rule.get("severity_on_pass", "pass")
            detail = _render_template(rule.get("detail_on_pass", "Check passed."), facts)
        findings.append({
            "rule_id": rule.get("id", "unknown"),
            "severity": sev,
            "title": rule.get("title", "Untitled rule"),
            "detail": detail,
            "source": rule.get("source", ""),
            "framework": rule.get("framework", "custom"),
            "tags": rule.get("tags", []),
        })
    return findings


def export_rules(fmt: str = "yaml", directories: list[Path] | None = None) -> str:
    """Export all loaded rules to YAML or JSON string."""
    rules = load_rules(directories)
    # Strip internal metadata
    clean = []
    for r in rules:
        c = {k: v for k, v in r.items() if not k.startswith("_")}
        clean.append(c)
    if fmt == "json":
        return json.dumps({"rules": clean}, indent=2, default=str)
    return yaml.dump({"rules": clean}, default_flow_style=False, sort_keys=False)


def import_rules(content: str, fmt: str = "yaml", filename: str = "imported.yaml") -> int:
    """Import rules from YAML/JSON string into custom rules directory. Returns count."""
    CUSTOM_RULES_DIR.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        data = json.loads(content)
    else:
        data = yaml.safe_load(content)
    if not data or "rules" not in data:
        raise ValueError("Input must contain a 'rules' key with a list of rule definitions.")
    count = len(data["rules"])
    outpath = CUSTOM_RULES_DIR / filename
    with open(outpath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return count


def get_rule_summary(findings: list[dict]) -> dict:
    """Summarize findings by framework and severity."""
    summary: dict[str, dict[str, int]] = {}
    for f in findings:
        fw = f["framework"]
        sev = f["severity"]
        if fw not in summary:
            summary[fw] = {}
        summary[fw][sev] = summary[fw].get(sev, 0) + 1
    return summary
