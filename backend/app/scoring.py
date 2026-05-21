"""
EpsiLint v3 — Privacy scoring and pyramid evaluation.
"""

from __future__ import annotations


def compute_score(facts: dict, findings: list[dict]) -> tuple[int, str]:
    """Compute a 0-100 privacy score from facts and rule findings."""
    s = 100

    # No recognized DP library — code cannot be meaningfully assessed
    if not facts.get("uses_tested_library") and facts.get("eps_count", 0) == 0:
        s -= 50

    # Epsilon penalties
    me = facts.get("max_eps", 0)
    if me > 10:
        s -= 35
    elif me > 3:
        s -= 20
    elif me > 1:
        s -= 8

    te = facts.get("total_epsilon", 0)
    if te > 10:
        s -= 15
    elif te > 5:
        s -= 8

    # Delta penalties
    for d in facts.get("deltas", []):
        n = facts.get("ctx_n", 10000)
        if d["value"] > 1.0 / n:
            s -= 15
        elif d["value"] > 1.0 / (n * n):
            s -= 5

    # Structural penalties
    if facts.get("has_loops"):
        s -= 12
    if not facts.get("has_explicit_bounds", True):
        s -= 8
    if facts.get("lib_count", 0) > 1:
        s -= 4
    if facts.get("ctx_unit") != "user":
        s -= 5
    if facts.get("ctx_trust") == "central":
        s -= 3
    if facts.get("has_gaussian_no_delta"):
        s -= 15
    if facts.get("ml_missing_bounds"):
        s -= 5
    if facts.get("ml_missing_data_norm"):
        s -= 3
    if facts.get("has_black_box_calls"):
        s -= 10
    if facts.get("has_noise_before_clip"):
        s -= 15

    # Bonuses from rule findings
    for f in findings:
        if f["severity"] == "pass":
            s += 1
        elif f["severity"] == "critical":
            s -= 3
        elif f["severity"] == "warning":
            s -= 1

    # Structural bonuses
    if facts.get("has_composition"):
        s += 3
    if facts.get("has_clamp"):
        s += 3
    if facts.get("gradient_clip_before_noise"):
        s += 2

    s = max(0, min(100, round(s)))
    if s >= 85:
        label = "Strong"
    elif s >= 70:
        label = "Moderate"
    elif s >= 50:
        label = "Weak"
    else:
        label = "Critical"
    return s, label


def build_pyramid(facts: dict) -> list[dict]:
    """Build the 9-layer DP Privacy Pyramid assessment."""
    layers = []

    # 1. Privacy parameters
    me = facts.get("max_eps", 0)
    mi = facts.get("min_eps", 0)
    if me > 0 and me < 1:
        st, d = "pass", f"ε range: {mi:.2f}–{me:.2f}. Strong parameters."
    elif me <= 10:
        st, d = "warn", f"ε range: {mi:.2f}–{me:.2f}. Moderate — review against data sensitivity."
    elif me > 0:
        st, d = "fail", f"ε range: {mi:.2f}–{me:.2f}. Weak — recalibrate."
    else:
        st, d = "na", "No epsilon detected."
    layers.append({"layer": "Privacy parameters (ε, δ)", "status": st, "detail": d})

    # 2. Unit of privacy
    u = facts.get("ctx_unit", "unknown")
    if u == "user":
        layers.append({"layer": "Unit of privacy", "status": "pass", "detail": "User-level (recommended strongest)."})
    elif u == "event":
        layers.append({"layer": "Unit of privacy", "status": "warn", "detail": "Event-level — weaker than user-level."})
    else:
        layers.append({"layer": "Unit of privacy", "status": "info", "detail": "Not specified."})

    # 3. Algorithm design
    tested = facts.get("uses_tested_library", False)
    clamped = facts.get("has_clamp", False)
    libs = ", ".join(facts.get("libraries", []))
    if tested and clamped:
        layers.append({"layer": "Algorithm design and correctness", "status": "pass", "detail": f"{libs}. Input clamping detected."})
    elif tested:
        layers.append({"layer": "Algorithm design and correctness", "status": "warn", "detail": f"{libs}. No explicit clamping — sensitivity may be unconstrained."})
    else:
        layers.append({"layer": "Algorithm design and correctness", "status": "fail", "detail": "No recognized DP library."})

    # 4. Utility assessment
    has_map = facts.get("has_privacy_map", False)
    has_comp = facts.get("has_composition", False)
    if has_map:
        layers.append({"layer": "Utility assessment", "status": "pass", "detail": "Privacy map detected."})
    elif has_comp:
        layers.append({"layer": "Utility assessment", "status": "pass", "detail": "Composition detected."})
    else:
        layers.append({"layer": "Utility assessment", "status": "info", "detail": "No explicit utility measurement. NIST recommends measuring accuracy."})

    # 5. Bias evaluation
    sens = facts.get("ctx_sensitivity", "medium")
    if sens == "high":
        layers.append({"layer": "Bias evaluation", "status": "warn", "detail": "High-sensitivity data: measure bias across demographics."})
    else:
        layers.append({"layer": "Bias evaluation", "status": "info", "detail": "Evaluate whether DP noise creates bias for minority subgroups."})

    # 6. Query model
    if facts.get("has_loops"):
        layers.append({"layer": "Query model and side channels", "status": "warn", "detail": f"Loop-based queries ({facts['loop_iterations']} iterations). Must track and cap budget."})
    else:
        layers.append({"layer": "Query model and side channels", "status": "info", "detail": "Determine batch vs interactive model. Mitigate side channels."})

    # 7. Trust model
    trust = facts.get("ctx_trust", "unknown")
    if trust in ("local", "distributed"):
        layers.append({"layer": "Trust model", "status": "pass", "detail": f"{trust.capitalize()} model selected."})
    elif trust == "central":
        layers.append({"layer": "Trust model", "status": "warn", "detail": "Central model selected."})
    else:
        layers.append({"layer": "Trust model", "status": "info", "detail": "Not specified."})

    # 8. Security
    layers.append({"layer": "Security and access control", "status": "info", "detail": "DP does NOT protect raw data at rest. Implement encryption and access controls independently."})

    # 9. Data collection
    if trust == "local":
        layers.append({"layer": "Data collection exposure", "status": "pass", "detail": "Local model minimizes collection exposure."})
    else:
        layers.append({"layer": "Data collection exposure", "status": "warn", "detail": "Raw data exposed during collection. Secure transport and minimize retention."})

    return layers


def build_recommendations(facts: dict, findings: list[dict]) -> list[dict]:
    """Generate actionable recommendations."""
    recs = []

    # From findings
    crit_hazards = [f for f in findings if f["severity"] == "critical" and f["framework"] == "nist"]
    if crit_hazards:
        ids = ", ".join(f["rule_id"] for f in crit_hazards)
        recs.append({"title": "Address NIST hazard failures", "detail": f"{len(crit_hazards)} hazard(s) failed: {ids}. Address before deployment.", "source": "NIST SP 800-226"})

    me = facts.get("max_eps", 0)
    if me > 1:
        recs.append({"title": "Reduce epsilon to below 1.0", "detail": f"Current max: {me}. NIST considers < 1 reasonable. Apple uses 2–8, Google ~9 at massive scale.", "source": "NIST SP 800-226 §2.2"})

    if facts.get("has_gaussian_no_delta"):
        recs.append({"title": "Add delta parameter to Gaussian mechanism", "detail": "Gaussian noise requires (ε,δ)-DP. Set delta <= 1/n².", "source": "NIST SP 800-226 §2.3"})

    if facts.get("uses_raw_mechanism") and not facts.get("has_sensitivity_param"):
        recs.append({"title": "Set explicit sensitivity for mechanisms", "detail": "Derive sensitivity from domain knowledge, not data inspection.", "source": "NIST SP 800-226 §3.4.1; DiffMu type system"})

    if facts.get("ctx_unit") != "user":
        recs.append({"title": "Consider user-level privacy", "detail": "NIST recommends user-level as default.", "source": "NIST SP 800-226 §2.4.2"})

    if facts.get("has_loops"):
        recs.append({"title": "Remove DP queries from loops", "detail": f"{facts['loop_iterations']} iterations multiply privacy cost. Pre-aggregate instead.", "source": "W3C DP Guidance §4.4"})

    if not facts.get("has_explicit_bounds", True) and facts.get("uses_tested_library"):
        recs.append({"title": "Specify explicit bounds", "detail": "Use domain knowledge for bounds, not data inspection.", "source": "NIST SP 800-226 §3.4.1"})

    if facts.get("lib_count", 0) > 1:
        recs.append({"title": "Unify privacy accounting", "detail": "Multiple libraries track budgets independently.", "source": "NIST SP 800-226 §2.5"})

    if not facts.get("has_composition") and facts.get("eps_count", 0) > 1:
        recs.append({"title": "Add formal composition", "detail": f"{facts['eps_count']} queries without explicit composition.", "source": "NIST SP 800-226 §3.4; DiffMu type inference"})

    if facts.get("uses_dpsgd") and not facts.get("has_random_label_test"):
        recs.append({"title": "Add memorization sanity check", "detail": "Train on random labels first. If model memorizes, ε is too loose.", "source": "van der Veen et al. 2018 §2.1"})

    recs.append({"title": "Add privacy unit tests", "detail": "Programmatically verify total ε stays within budget.", "source": "NIST SP 800-226 §3"})

    return recs
