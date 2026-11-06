# NeuroEdge website — private review candidate

Static HTML/CSS/JS. The hero is a conceptual illustration, and the SVG architecture is a deliberately editable technical diagram. Solid paths describe the original simulation-verified digital input-to-logit pipeline. Dashed paths describe **software-only supervised synthetic feedback**; they do not represent a clinical, patient-operated or implanted loop.

Preview:

```bash
python3 -m http.server 8080 --directory website
# http://localhost:8080
```

The interactive replay reads `demo/closed_loop_replay.json`, generated with `PYTHONPATH=. python3 scripts/run_closed_loop_replay.py --out website/demo/closed_loop_replay.json`. Its per-trial numerical values are generated from a deterministic non-learned Python fixture; they are not clinical performance. No external analytics, forms, credentials, or remote services are used.

Do not deploy before checking live desktop/mobile rendering, fresh acceptance on the exact consolidated source, rights to publish the code and all scientific statements. GitHub Pages has been configured to use `website/` but deployment is **not** requested.

The page contains `robots=noindex,nofollow` while this is a private candidate; intentional public deployment would require a separate rights/evidence review and deliberate change.
