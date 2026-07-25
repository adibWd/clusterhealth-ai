# Design Note — Report Card & Threshold Tuner

**Team NephroAI · ClusterHealth AI · AI Innovation Hackathon 2026, DIU CSE**
**Final Round — Additional Challenge Task**

## 1. The problem, restated

ClusterHealth AI predicts failures and explains them with SHAP, but the
alert threshold was fixed in code. A strict threshold under-alerts (misses
real faults); a loose threshold over-alerts (buries the operator in false
alarms). Different clusters, workloads, and risk tolerances want different
balance points, and the system gave nobody a way to see — let alone
choose — that trade-off.

## 2. Design goals, in priority order

1. **Never re-score inconsistently with the rest of the product.** The
   Report Card must use the exact same trained XGBoost + BiGRU-Attention
   ensemble and the exact same 0.6/0.4 weighting as the main dashboard, or
   its counts would quietly disagree with what the operator sees elsewhere.
2. **No training inside the session.** The brief is explicit about this.
   The module only calls `HybridModelService.score(..., explain=False)` on
   already-trained artifacts — nothing is fit, updated, or fine-tuned.
3. **The slider must feel instant in a live demo.** Re-running inference on
   every pixel of drag would be both slow and unnecessary — the underlying
   telemetry doesn't change, only the decision rule does.
4. **Zero blast radius.** No existing endpoint, model file, or dashboard
   panel should need to change for this feature to exist.

## 3. Key decisions

**Score once, cache, then it's just arithmetic.**
At API startup, every labeled sliding window in `data/telemetry.csv`
(26,100 windows across 300 devices, reusing `features.build_windows`,
unchanged) is scored once and the resulting `(probability, ground_truth)`
arrays are cached in `app.state`. Every subsequent threshold change is a
single NumPy comparison (`probabilities >= threshold`) against that cache
— sub-millisecond, regardless of how many times a judge drags the slider.

**A new router, not new logic bolted into `api.py`.**
`report_card.py` is a self-contained `APIRouter` mounted with
`app.include_router(report_card.router)`. `api.py` gained two lines (an
import and the mount) plus a startup cache-warm call. This was a deliberate
choice to keep the "quality of integration with the existing system" clean
and reviewable — a diff of `api.py` shows exactly what changed and why.

**Ground truth comes from the dataset's own injected labels.**
`generate_data.py` already writes a `failure_within_horizon` label for
every timestep (1 for the ~2-hour window before/during an injected
failure). The Report Card reuses that label as-is — no new labeling
scheme, no synthetic ground truth invented for this task.

**Why F2 (recall-weighted), not plain accuracy or F1, for the suggestion.**
In this domain, a missed fault (unplanned downtime, possible data loss, an
emergency page) is more expensive than one extra false-alarm investigation.
F-beta with β=2 formalizes that: recall is weighted roughly 4× precision
when scoring candidate thresholds. This is a defensible, named,
industry-standard metric rather than a hand-tuned heuristic — and it is
stated openly in the reason string returned to the operator, not hidden in
code.

**Why a grid sweep (5%–95%, 1% steps) instead of a closed-form optimum.**
A full sweep is simple, deterministic, cheap (91 candidates × O(n) counts,
trivial at this data size), and easy to defend in Q&A: "we tried every
threshold from 5% to 95% and kept the one with the best recall-weighted
score." No solver, no approximation to explain.

**Tie-breaking is explicit, not accidental.**
The dataset's clean, injected failure signatures make the ensemble's
probabilities saturate near 0 and 1 (the same calibration caveat the main
README already flags in Section 4). That means several thresholds can tie
on F2. Ties are broken by (a) fewest false alarms, then (b) closest to the
midpoint of the sweep range — a stable rule stated in code comments, not an
arbitrary "first match wins."

**The tuner is intentionally decoupled from the Live/Static toggle.**
The brief asks for the Report Card and Tuner to work "on the same stored
data." Tying it to the live 8-device simulator would mean its counts drift
every 5 seconds and stop being comparable across a demo. It loads once per
session against the stored run and only recomputes when the operator moves
the threshold.

## 4. What was deliberately left out (and why)

- **No new Docker service, no Prometheus/Grafana wiring.** The task is
  scoped to inference + counting over already-stored data; adding
  infrastructure would have increased integration risk for no requirement
  gained.
- **No retraining or threshold persistence across restarts.** The brief
  caps scope at "no long model training inside the session" and a demo-time
  control; the cache is intentionally in-memory and rebuilt at startup.
- **No per-device breakdown in the Report Card table.** The brief asks for
  one small table of the three counts; a per-device drill-down was judged
  as scope creep against the 2.5-hour build window and can be added later
  (see Section 5).

## 5. If we had another hour

- Add a small ROC/PR curve visualization next to the tuner so the operator
  sees the *shape* of the trade-off, not just one point on it.
- Persist the operator's chosen threshold (e.g. to `localStorage` outside
  the artifact sandbox, or a tiny `/reportcard/threshold` `PUT` endpoint)
  so it survives a page reload.
- Calibrate the ensemble (Platt scaling / isotonic regression) so the
  probability axis — and therefore the slider — spreads out more
  realistically instead of saturating near 0% and 100%.
