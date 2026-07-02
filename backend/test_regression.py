"""EpsiLint v3.1 regression suite — encodes the validation findings as asserts."""
import sys, math
sys.path.insert(0, ".")
from app.analyzer import extract_facts
from app.rule_engine import load_rules, evaluate_rules
from app.scoring import compute_score

RULES = load_rules()

def run(code, ctx=None):
    facts = extract_facts(code, ctx or {})
    findings = evaluate_rules(facts, RULES)
    # add per-line eps/delta like main.py (needed for score anchor on eps)
    n = facts["ctx_n"]; dt = 1.0/(n*n)
    for e in facts["epsilons"]:
        v=e["value"]; sev="critical" if v>10 else "warning" if v>3 else "info" if v>1 else "pass"
        findings.append({"severity":sev,"title":f"eps{v}","framework":"code","rule_id":"eps-line"})
    for d in facts["deltas"]:
        v=d["value"]
        if v>1.0/n: findings.append({"severity":"critical","title":"d","framework":"code","rule_id":"delta-line"})
    findings=[f for f in findings if f["severity"]!="skip"]
    score,label=compute_score(facts,findings)
    return facts, findings, score, label

def crit_titles(findings): return [f["title"] for f in findings if f["severity"]=="critical"]
def has_crit(findings, sub): return any(sub in t for t in crit_titles(findings))
def has_warn(findings, sub): return any(sub in f["title"] for f in findings if f["severity"]=="warning")

PASS=0; FAIL=0
def check(name, cond, extra=""):
    global PASS,FAIL
    print(("  PASS " if cond else "  FAIL ")+name+("" if cond else "  <-- "+extra))
    PASS+=cond; FAIL+=(not cond)

# H — crash-safe
try:
    f,fd,sc,lb = run('from diffprivlib.mechanisms import Laplace\n# swept epsilon = 0.1-1.0\nm = Laplace(epsilon=0.5, sensitivity=1.0)')
    check("H no-crash on comment range; eps=[0.5]", f["eps_values"]==[0.5], str(f["eps_values"]))
except Exception as e:
    check("H no-crash on comment range", False, repr(e))

# D — Gaussian no delta: critical AND score in Critical band
f,fd,sc,lb = run('from diffprivlib.mechanisms import Gaussian\ng = Gaussian(epsilon=0.5, sensitivity=1.0)\nout=g.randomise(x)')
check("D Gaussian-no-delta raises critical", has_crit(fd,"requires delta"))
check("D score anchored to Critical band (<=49)", sc<=49, f"score={sc} {lb}")

# J — Adam epsilon ignored
f,fd,sc,lb = run('epsilon = 1e-8  # numerical stability\nloss = x/(d+epsilon)')
check("J optimizer epsilon NOT counted", f["eps_count"]==0, str(f["eps_values"]))
check("J non-DP code not 'Strong'", lb!="Strong", f"{sc} {lb}")

# F — pandas sample not a DP mechanism
f,fd,sc,lb = run('import pandas as pd\ntrain = df.sample(frac=0.8)')
check("F pandas .sample not flagged as Subsampling", "Subsampling" not in f["mech_names"])

# G — unrelated C=/clip() not DP-SGD clipping
f,fd,sc,lb = run('C = 1.0\ntemp = clip(logits,-1,1)\nout=C*temp')
check("G no false clipping-bound", f["has_clipping_bound"]==False)

# I — BoundedMean missing bounds flagged
f,fd,sc,lb = run('import pydp as dp\nfrom pydp.algorithms.laplacian import BoundedMean\nm=BoundedMean(epsilon=0.5)\nm.quick_result(x)')
check("I BoundedMean missing bounds detected", f["has_explicit_bounds"]==False)
check("I missing-bounds warning emitted", has_warn(fd,"bounds") or has_warn(fd,"Bounds"))

# K/L — ordering
f,fd,sc,lb = run('noisy = grads + tf.random.normal(g.shape)*noise_multiplier\nclipped = tf.clip_by_norm(noisy, max_grad_norm)', {"training_mode":"dpsgd"})
check("K noise-before-clip detected", f["has_noise_before_clip"]==True)
f,fd,sc,lb = run('clipped = tf.clip_by_norm(grads, max_grad_norm)\nnoisy = clipped + tf.random.normal(clipped.shape)*noise_multiplier', {"training_mode":"dpsgd"})
check("L clip-before-noise credited (naming-robust)", f["gradient_clip_before_noise"]==True)

# B — canonical Opacus: no spurious loop-critical, delta 1e-5 not a warning
opacus = '''from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(module=model, optimizer=optimizer, data_loader=data_loader, noise_multiplier=1.1, max_grad_norm=1.0)
for epoch in range(10):
    train(model, data_loader, optimizer)
eps = privacy_engine.get_epsilon(delta=1e-5)'''
f,fd,sc,lb = run(opacus, {"training_mode":"dpsgd","unit":"user","n":60000})
check("B epoch loop NOT flagged as DP-query loop", f["has_dp_query_loop"]==False)
check("B no loop-budget critical", not has_crit(fd,"exhaustion") and not has_crit(fd,"budget"))
check("B delta=1e-5 @n=60000 not a warning", not has_warn(fd,"delta") and not has_warn(fd,"1/n"))
check("B accountant epsilon recognized", f["has_computed_epsilon"]==True)

# Composition off-by-one
f,fd,sc,lb = run('from pydp.algorithms.laplacian import Count\nfor i in range(20):\n    q=Count(epsilon=1.0)\n    q.quick_result(x)')
check("Loop composition total == 20 (not 21)", abs(f["total_epsilon"]-20.0)<1e-9, str(f["total_epsilon"]))

# Advanced composition still correct
f,fd,sc,lb = run("from diffprivlib.mechanisms import Laplace\n"+"\n".join(f"m{i}=Laplace(epsilon=0.1, sensitivity=1.0)" for i in range(100)))
k=100;e=0.1;d=1e-5; adv=e*math.sqrt(2*k*math.log(1/d))+k*e*(math.exp(e)-1)
check("Advanced composition matches textbook (~5.85)", abs(f["advanced_epsilon"]-adv)<1e-6, str(f["advanced_epsilon"]))

# Float-laplace suppressed for OpenDP
f,fd,sc,lb = run('import opendp.prelude as dp\ndp.enable_features("contrib")\nx = dp.t.make_clamp(bounds=(0.0,100.0)) >> dp.m.then_laplace(scale=1.0)')
check("OpenDP not flagged for float-Laplace", f["float_laplace_vulnerable"]==False)
check("OpenDP correct pipeline not capped Critical", lb in ("Strong","Moderate"), f"{sc} {lb}")

# Title/severity labels fixed
f,fd,sc,lb = run("x = 1+1")
check("No-library critical is correctly titled", has_crit(fd,"No recognized DP library"))
check("Deniability critical correctly titled", has_crit(fd,"No perturbation"))


# OpenDP then_gaussian must NOT be flagged as missing delta
f,fd,sc,lb = run('import opendp.prelude as dp\ndp.enable_features("contrib")\nx = c >> dp.m.then_gaussian(scale=2.0)')
check("OpenDP then_gaussian not flagged no-delta", not has_crit(fd,"requires delta"))
# Opacus make_private must NOT be flagged noise-before-clip
opa='''from opacus import PrivacyEngine
m,o,d = PrivacyEngine().make_private(module=m, optimizer=o, data_loader=d,
    noise_multiplier=1.1,
    max_grad_norm=1.0)'''
f,fd,sc,lb = run(opa, {"training_mode":"dpsgd"})
check("Opacus make_private not flagged noise-before-clip", f["has_noise_before_clip"]==False)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
