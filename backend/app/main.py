"""
EpsiLint v3 — FastAPI Backend Server.

A Linter for Catching Common Mistakes in Differential Privacy Code.
YAML/JSON-driven policy engine.
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from .models import AnalysisRequest, AnalysisResult
from .analyzer import extract_facts
from .rule_engine import load_rules, evaluate_rules, export_rules, import_rules, get_rule_summary
from .scoring import compute_score, build_pyramid, build_recommendations

app = FastAPI(
    title="EpsiLint: A Linter for Catching Common Mistakes in Differential Privacy Code",
    description=(
        "Analyze Python differential privacy code against NIST SP 800-226, "
        "W3C DP Guidance, van der Veen et al. (2018) practical DP tools, "
        "and DiffMu type-system-inspired sensitivity rules."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Built-in examples ──
EXAMPLES = {
    "pydp": {
        "label": "PyDP example",
        "code": (
            'import pydp as dp\n'
            'from pydp.algorithms.laplacian import BoundedMean, BoundedSum, Count\n\n'
            'epsilon = 0.5\ndelta = 1e-5\n\n'
            'mean_age = BoundedMean(epsilon=0.5, lower_bound=18, upper_bound=90, dtype="float")\n'
            'mean_age.quick_result(ages)\n\n'
            'sum_income = BoundedSum(epsilon=0.3, lower_bound=0, upper_bound=500000, dtype="float")\n'
            'sum_income.quick_result(incomes)\n\n'
            'count_users = Count(epsilon=0.2, dtype="int")\n'
            'count_users.quick_result(users)'
        ),
    },
    "opendp": {
        "label": "OpenDP example",
        "code": (
            'import opendp.prelude as dp\n'
            'dp.enable_features("contrib", "honest-but-curious")\n\n'
            'bounded = (\n'
            '    dp.t.make_clamp(bounds=(0.0, 100.0)) >>\n'
            '    dp.t.make_bounded_resize(size=1000, bounds=(0.0, 100.0), constant=50.0) >>\n'
            '    dp.t.make_sized_bounded_mean(size=1000, bounds=(0.0, 100.0))\n'
            ')\n\n'
            'noisy_mean = bounded >> dp.m.then_laplace(scale=1.5)\n'
            'noisy_sum = bounded >> dp.m.then_gaussian(scale=2.0)\n\n'
            'composed = dp.c.make_basic_composition([noisy_mean, noisy_sum])'
        ),
    },
    "diffprivlib": {
        "label": "diffprivlib example",
        "code": (
            'from diffprivlib.models import GaussianNB, LogisticRegression\n'
            'from diffprivlib.mechanisms import Laplace, Gaussian\n'
            'import numpy as np\n\n'
            'clf = GaussianNB(epsilon=1.0, bounds=[(0, 1)] * 4)\n'
            'clf.fit(X_train, y_train)\n\n'
            'lr = LogisticRegression(epsilon=2.0, data_norm=5.0)\n'
            'lr.fit(X_train, y_train)\n\n'
            'lap = Laplace(epsilon=0.5, sensitivity=1.0)\n'
            'private_count = lap.randomise(true_count)\n\n'
            'gauss = Gaussian(epsilon=0.8, delta=1e-5, sensitivity=1.0)\n'
            'private_mean = gauss.randomise(true_mean)'
        ),
    },
    "weak": {
        "label": "Weak privacy (bad example)",
        "code": (
            'import pydp as dp\n'
            'from pydp.algorithms.laplacian import BoundedMean, BoundedSum, Count\n'
            'from diffprivlib.models import LogisticRegression\n\n'
            'mean_salary = BoundedMean(epsilon=50.0, lower_bound=0, upper_bound=1000000, dtype="float")\n'
            'result1 = mean_salary.quick_result(salaries)\n\n'
            'sum_vals = BoundedSum(epsilon=10, dtype="float")\n'
            'sum_vals.quick_result(values)\n\n'
            'for i in range(20):\n'
            '    q = Count(epsilon=1.0, dtype="int")\n'
            '    q.quick_result(sensitive_data)\n\n'
            'lr = LogisticRegression(epsilon=100, data_norm=5.0)\n'
            'lr.fit(X_train, y_train)\n\n'
            'from diffprivlib.mechanisms import Gaussian\n'
            'g = Gaussian(epsilon=5, delta=0.1, sensitivity=1.0)\n'
            'result = g.randomise(value)'
        ),
    },
    "dpsgd": {
        "label": "DP-SGD training",
        "code": (
            'import opacus\n'
            'from opacus import PrivacyEngine\n'
            'import torch\n\n'
            'model = Net()\n'
            'optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n'
            'privacy_engine = PrivacyEngine()\n\n'
            'model, optimizer, train_loader = privacy_engine.make_private(\n'
            '    module=model,\n'
            '    optimizer=optimizer,\n'
            '    data_loader=train_loader,\n'
            '    noise_multiplier=1.1,\n'
            '    max_grad_norm=1.0,\n'
            '    batch_size=64,\n'
            ')\n\n'
            'epsilon = privacy_engine.get_epsilon(delta=1e-5)\n'
            'print(f"Achieved epsilon: {epsilon}")'
        ),
    },
}


@app.get("/api/examples")
def get_examples():
    """Return all built-in example snippets."""
    return EXAMPLES


@app.post("/api/analyze")
def analyze(req: AnalysisRequest) -> dict:
    """Run full privacy analysis on submitted code."""
    ctx = req.context.model_dump()
    facts = extract_facts(req.code, ctx)
    rules = load_rules()
    findings = evaluate_rules(facts, rules)

    # Per-epsilon and per-delta line-level findings
    per_line = []
    for e in facts["epsilons"]:
        v = e["value"]
        if v > 10:
            per_line.append({"rule_id": "eps-line", "severity": "critical", "title": f"Excessive ε ({v}) at line {e['line']}", "detail": f"Provides virtually no privacy. Reduce to < 1.", "source": "NIST SP 800-226 §2.2", "framework": "code", "tags": ["epsilon"]})
        elif v > 3:
            per_line.append({"rule_id": "eps-line", "severity": "warning", "title": f"High ε ({v}) at line {e['line']}", "detail": "Weak DP. Attacker can distinguish neighboring datasets.", "source": "NIST SP 800-226 §2.2", "framework": "code", "tags": ["epsilon"]})
        elif v > 1:
            per_line.append({"rule_id": "eps-line", "severity": "info", "title": f"Moderate ε ({v}) at line {e['line']}", "detail": "Within deployed range but permits noticeable leakage per query.", "source": "NIST SP 800-226 §2.2", "framework": "code", "tags": ["epsilon"]})
        else:
            per_line.append({"rule_id": "eps-line", "severity": "pass", "title": f"Strong ε ({v}) at line {e['line']}", "detail": "Strong per-query privacy protection.", "source": "NIST SP 800-226 §2.2", "framework": "code", "tags": ["epsilon"]})

    n = facts["ctx_n"]
    d_thresh = 1.0 / (n * n)
    for d in facts["deltas"]:
        v = d["value"]
        if v > 1.0 / n:
            per_line.append({"rule_id": "delta-line", "severity": "critical", "title": f"δ {v} at line {d['line']} — catastrophic", "detail": f"Exceeds 1/n. Set δ ≤ 1/n² = {d_thresh:.1e}.", "source": "NIST SP 800-226 §2.3", "framework": "code", "tags": ["delta"]})
        elif v > d_thresh:
            per_line.append({"rule_id": "delta-line", "severity": "warning", "title": f"δ {v:.1e} above 1/n² at line {d['line']}", "detail": f"NIST recommends δ ≤ {d_thresh:.1e} for n={n}.", "source": "NIST SP 800-226 §2.3", "framework": "code", "tags": ["delta"]})

    # Filter out 'skip' severities
    all_findings = [f for f in findings if f["severity"] != "skip"]
    all_findings.extend(per_line)
    # Sort: critical first, then warning, info, pass
    sev_order = {"critical": 0, "warning": 1, "info": 2, "pass": 3}
    all_findings.sort(key=lambda f: sev_order.get(f["severity"], 4))

    score, label = compute_score(facts, all_findings)
    pyramid = build_pyramid(facts)
    recs = build_recommendations(facts, all_findings)
    summary = get_rule_summary(all_findings)

    # Composition table
    comp_entries = []
    cum = 0
    for e in facts["epsilons"]:
        cum += e["value"]
        comp_entries.append({
            "query": e["context"][:50],
            "library": facts["libraries"][0] if facts["libraries"] else "—",
            "epsilon": e["value"],
            "cumulative": round(cum, 4),
            "line": e["line"],
        })
    if facts["has_loops"] and facts["eps_values"]:
        loop_eps = facts["eps_values"][0]
        cum += facts["loop_iterations"] * loop_eps
        comp_entries.append({
            "query": f"Loop (x{facts['loop_iterations']})",
            "library": "—",
            "epsilon": loop_eps * facts["loop_iterations"],
            "cumulative": round(cum, 4),
            "line": 0,
        })

    return {
        "score": score,
        "score_label": label,
        "libraries": facts["libraries"],
        "epsilons": facts["epsilons"],
        "deltas": facts["deltas"],
        "mechanisms": facts["mechanisms"],
        "total_epsilon": round(facts["total_epsilon"], 4),
        "advanced_epsilon": round(facts["advanced_epsilon"], 4),
        "findings": all_findings,
        "pyramid": pyramid,
        "composition": comp_entries,
        "recommendations": recs,
        "rule_summary": summary,
    }


@app.get("/api/rules/export")
def rules_export(fmt: str = Query("yaml", enum=["yaml", "json"])):
    """Export all loaded rules as YAML or JSON."""
    content = export_rules(fmt)
    media = "application/x-yaml" if fmt == "yaml" else "application/json"
    return PlainTextResponse(content=content, media_type=media)


@app.post("/api/rules/import")
async def rules_import(
    file: UploadFile = File(...),
    fmt: str = Query("yaml", enum=["yaml", "json"]),
):
    """Import custom rules from a YAML or JSON file."""
    content = (await file.read()).decode("utf-8")
    try:
        count = import_rules(content, fmt, filename=file.filename or "imported.yaml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"imported": count, "filename": file.filename}


@app.get("/api/rules")
def list_rules():
    """List all currently loaded rules."""
    rules = load_rules()
    return {"count": len(rules), "rules": rules}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "3.0.0"}
