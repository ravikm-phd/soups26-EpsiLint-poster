"""
EpsiLint v3 — Static Analysis Engine for Differential Privacy Code.

Extracts structural facts from Python DP code. The rule engine
(rule_engine.py) evaluates YAML-defined policies against these facts.
"""

from __future__ import annotations
import re, math
from typing import Any


def extract_facts(code: str, ctx: dict) -> dict:
    """Parse Python DP code and return a fact dictionary used by rule evaluation."""
    lines = code.split("\n")
    facts: dict[str, Any] = {
        "code": code,
        "lines": lines,
        "line_count": len(lines),
        # Context
        "ctx_n": ctx.get("n", 10000),
        "ctx_trust": ctx.get("trust", "unknown"),
        "ctx_sensitivity": ctx.get("sensitivity", "medium"),
        "ctx_unit": ctx.get("unit", "event"),
        "ctx_training_mode": ctx.get("training_mode", "none"),
        "ctx_batch_size": ctx.get("batch_size"),
        "ctx_learning_rate": ctx.get("learning_rate"),
    }

    # ── Library detection ──
    libs = []
    if re.search(r"import\s+pydp|from\s+pydp", code):
        libs.append("PyDP")
    if re.search(r"import\s+opendp|from\s+opendp", code):
        libs.append("OpenDP")
    if re.search(r"import\s+diffprivlib|from\s+diffprivlib", code):
        libs.append("diffprivlib")
    if re.search(r"import\s+tensorflow_privacy|from\s+tensorflow_privacy", code):
        libs.append("tensorflow_privacy")
    if re.search(r"import\s+opacus|from\s+opacus", code):
        libs.append("Opacus")
    facts["libraries"] = libs
    facts["lib_count"] = len(libs)
    facts["uses_tested_library"] = len(libs) > 0

    # ── Epsilon extraction ──
    eps_list = []
    for m in re.finditer(r"epsilon\s*[=:]\s*([\d.eE\-+]+)", code):
        val = float(m.group(1))
        if val > 0:
            ln = code[:m.start()].count("\n") + 1
            ctx_line = lines[ln - 1].strip() if ln <= len(lines) else ""
            eps_list.append({"value": val, "line": ln, "context": ctx_line})
    facts["epsilons"] = eps_list
    facts["eps_values"] = [e["value"] for e in eps_list]
    facts["eps_count"] = len(eps_list)
    facts["max_eps"] = max((e["value"] for e in eps_list), default=0)
    facts["min_eps"] = min((e["value"] for e in eps_list), default=0)

    # ── Delta extraction ──
    delta_list = []
    for m in re.finditer(r"delta\s*[=:]\s*([\d.eE\-+]+)", code):
        val = float(m.group(1))
        ln = code[:m.start()].count("\n") + 1
        delta_list.append({"value": val, "line": ln})
    facts["deltas"] = delta_list
    facts["delta_values"] = [d["value"] for d in delta_list]
    facts["max_delta"] = max((d["value"] for d in delta_list), default=None)
    facts["delta_count"] = len(delta_list)
    facts["unique_delta_count"] = len(set(d["value"] for d in delta_list))
    n = facts["ctx_n"]
    facts["delta_threshold"] = 1.0 / (n * n) if n > 0 else 1e-10

    # ── Mechanism detection ──
    mech_patterns = [
        (r"BoundedMean",              "Bounded mean",          "PyDP"),
        (r"BoundedSum",               "Bounded sum",           "PyDP"),
        (r"Count\(",                  "Count",                 "PyDP"),
        (r"then_laplace",             "Laplace mechanism",     "OpenDP"),
        (r"then_gaussian",            "Gaussian mechanism",    "OpenDP"),
        (r"make_basic_composition",   "Basic composition",     "OpenDP"),
        (r"make_sequential_composition", "Sequential composition", "OpenDP"),
        (r"make_clamp",               "Clamping transform",    "OpenDP"),
        (r"make_bounded_resize",      "Bounded resize",        "OpenDP"),
        (r"Laplace\(",               "Laplace mechanism",     "diffprivlib"),
        (r"Gaussian\(",              "Gaussian mechanism",    "diffprivlib"),
        (r"GaussianNB",              "Gaussian Naive Bayes",  "diffprivlib"),
        (r"LogisticRegression",      "Logistic regression",   "diffprivlib"),
        (r"DPSGD|DPOptimizer|PrivacyEngine", "DP-SGD optimizer", "tf_privacy/Opacus"),
        (r"above_threshold",          "Above threshold",       "DiffMu-inspired"),
        (r"sample\s*\(",             "Subsampling",           "general"),
    ]
    mechs = []
    query_count = 0
    for pat, name, lib in mech_patterns:
        matches = re.findall(pat, code)
        if matches:
            mechs.append({"name": name, "library": lib, "count": len(matches)})
            query_count += len(matches)
    facts["mechanisms"] = mechs
    facts["mech_names"] = [m["name"] for m in mechs]
    facts["query_count"] = query_count

    # ── Loop detection ──
    loop_match = re.search(r"for\s+\w+\s+in\s+range\((\d+)\)", code)
    facts["has_loops"] = loop_match is not None
    facts["loop_iterations"] = int(loop_match.group(1)) if loop_match else 0
    if facts["has_loops"]:
        query_count += facts["loop_iterations"] - 1

    # ── Structural checks ──
    facts["has_composition"] = bool(re.search(r"make_basic_composition|make_sequential_composition", code))
    facts["has_clamp"] = bool(re.search(r"make_clamp", code))
    facts["has_bounded_resize"] = bool(re.search(r"make_bounded_resize|bounded_resize", code))
    facts["has_privacy_map"] = bool(re.search(r"\.map\(", code))
    facts["has_enable_features"] = bool(re.search(r"enable_features", code))
    facts["uses_opendp"] = "OpenDP" in libs
    facts["uses_gaussian"] = bool(re.search(r"Gaussian\(|then_gaussian", code))
    facts["has_gaussian_no_delta"] = facts["uses_gaussian"] and len(delta_list) == 0
    facts["has_sensitivity_param"] = bool(re.search(r"sensitivity\s*[=:]\s*[\d.]", code))
    facts["uses_raw_mechanism"] = bool(re.search(r"Laplace\(|Gaussian\(", code))
    facts["uses_laplace"] = bool(re.search(r"Laplace\(|then_laplace", code))
    facts["has_float_type"] = bool(re.search(r"dtype\s*=\s*[\"']float|T=float", code))

    # Bounds on BoundedSum
    bs_match = re.search(r"BoundedSum\([^)]*\)", code)
    facts["has_explicit_bounds"] = True
    if bs_match and not re.search(r"lower_bound", bs_match.group(0)):
        facts["has_explicit_bounds"] = False

    # ML model bounds (GaussianNB needs bounds, LogisticRegression needs data_norm)
    facts["ml_missing_bounds"] = False
    for m in re.finditer(r"GaussianNB\(([^)]*)\)", code):
        if "bounds" not in m.group(1):
            facts["ml_missing_bounds"] = True
    facts["ml_missing_data_norm"] = False
    for m in re.finditer(r"LogisticRegression\(([^)]*)\)", code):
        if "data_norm" not in m.group(1):
            facts["ml_missing_data_norm"] = True

    # ── DP-SGD specific checks (van der Veen paper) ──
    facts["uses_dpsgd"] = bool(re.search(
        r"DPSGD|DPOptimizer|PrivacyEngine|dp_sgd|dpsgd|make_gaussian_mechanism.*gradient",
        code, re.IGNORECASE
    ))
    facts["has_clipping_bound"] = bool(re.search(
        r"max_grad_norm|clipping_bound|clip_bound|max_norm|C\s*=\s*[\d.]|clip!\(|clip\(",
        code
    ))
    facts["has_noise_multiplier"] = bool(re.search(
        r"noise_multiplier|noise_scale|sigma\s*=\s*[\d.]",
        code
    ))
    facts["has_batch_size"] = bool(re.search(
        r"batch_size\s*=\s*\d+",
        code
    ))
    batch_match = re.search(r"batch_size\s*=\s*(\d+)", code)
    facts["detected_batch_size"] = int(batch_match.group(1)) if batch_match else None
    facts["has_lr_scaling"] = bool(re.search(
        r"learning_rate.*batch|lr.*batch|batch.*lr|batch.*learning_rate|scale.*lr|lr.*scale",
        code, re.IGNORECASE
    ))
    facts["has_random_label_test"] = bool(re.search(
        r"random.*label|shuffle.*label|permut.*label|label.*random|memorization.*test",
        code, re.IGNORECASE
    ))
    facts["has_moments_accountant"] = bool(re.search(
        r"moments_accountant|rdp_accountant|privacy_accountant|compute_epsilon|get_epsilon",
        code, re.IGNORECASE
    ))

    # ── DiffMu-inspired sensitivity tracking ──
    facts["has_sensitivity_annotation"] = bool(re.search(
        r"sensitivity\s*=|@sensitivity|sens\s*=\s*[\d.]|\.sensitivity",
        code
    ))
    facts["has_black_box_calls"] = bool(re.search(
        r"BlackBox|black_box|untrusted_function|external_call|subprocess|exec\(|eval\(",
        code
    ))
    facts["has_gradient_clipping"] = bool(re.search(
        r"clip_grad|clip!\(|clip\(.*norm|max_grad_norm|per_sample_clip",
        code
    ))
    facts["gradient_clip_before_noise"] = False
    clip_line = -1
    noise_line = -1
    for i, line in enumerate(lines):
        if re.search(r"clip|max_grad_norm|per_sample_clip", line, re.IGNORECASE):
            clip_line = i
        if re.search(r"gaussian_mechanism|laplacian_mechanism|add_noise|noise_multiplier", line, re.IGNORECASE):
            if noise_line < 0:
                noise_line = i
    # If both appear within the same function call (within 10 lines), don't flag ordering
    same_call = (clip_line >= 0 and noise_line >= 0 and abs(clip_line - noise_line) <= 10)
    if same_call:
        # Check if they're inside the same parenthesized call — look back further
        # to capture the function call that wraps both parameters
        window_start = max(0, min(clip_line, noise_line) - 5)
        window_end = max(clip_line, noise_line) + 2
        window = "\n".join(lines[window_start:window_end])
        if re.search(r"make_private|PrivacyEngine|DPOptimizer", window, re.IGNORECASE):
            same_call = True  # Both in a framework call — ordering is handled internally
        else:
            same_call = False
    facts["gradient_clip_before_noise"] = (clip_line >= 0 and noise_line >= 0 and clip_line < noise_line and not same_call)
    facts["has_noise_before_clip"] = (clip_line >= 0 and noise_line >= 0 and noise_line < clip_line and not same_call)

    # Post-processing near DP output
    dp_output_pat = re.compile(
        r"quick_result|randomise|randomize|\.release|noisy_|private_|dp_|then_laplace|then_gaussian|make_basic_composition"
    )
    facts["has_post_processing_near_dp"] = False
    if dp_output_pat.search(code) and not re.search(r"make_clamp", code):
        for i, line in enumerate(lines):
            if re.search(r"clip|max\s*\(\s*0|round\s*\(|abs\s*\(|\.astype\s*\(int\)", line):
                window = "\n".join(lines[max(0, i-3):i+2])
                if dp_output_pat.search(window):
                    facts["has_post_processing_near_dp"] = True
                    break

    # Raw data access after DP query
    facts["raw_data_after_dp"] = False
    dp_query_line = -1
    for i, line in enumerate(lines):
        if re.search(r"quick_result|randomise|randomize|\.release\(|\.fit\(", line):
            dp_query_line = i
        if dp_query_line >= 0 and i > dp_query_line:
            if re.search(r"print\s*\(.*(?:salaries|ages|incomes|sensitive|X_train|y_train|raw|original)", line):
                facts["raw_data_after_dp"] = True
                break

    # ── Composition math ──
    total_eps = sum(facts["eps_values"])
    if facts["has_loops"] and facts["eps_values"]:
        total_eps += facts["loop_iterations"] * facts["eps_values"][0]
    facts["total_epsilon"] = total_eps

    k = len(eps_list) + (facts["loop_iterations"] if facts["has_loops"] else 0)
    if k > 0 and total_eps > 0:
        avg_eps = total_eps / k
        delta_for_adv = 1e-5
        adv = min(total_eps, avg_eps * math.sqrt(2 * k * math.log(1.0 / delta_for_adv)) + k * avg_eps * (math.exp(avg_eps) - 1))
    else:
        adv = 0
    facts["advanced_epsilon"] = adv

    return facts
