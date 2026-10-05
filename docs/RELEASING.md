# Releasing checkpoints

The heads under `checkpoints/` are exported from training runs (`runs/`, not tracked) together with their model cards.
Every number in a card is read from evaluation outputs, so cards, README and `docs/RESULTS.md` come from the same files.

1. Train and evaluate (README, *Reproduce*): `eval.evaluate`, `eval.typed_decisions`, `eval.none_eval`,
   `scripts/bench_quant.py` (accuracy) and `scripts/bench_quant.py --no-acc` once per configuration (memory and
   latency, fresh process each), `eval.overlap_audit`.
2. Export: `.venv/bin/python scripts/export_checkpoints.py` (releases are listed in `RELEASES` at the top of the
   script). Shards stay under 50 MB so the repository needs no Git LFS.
3. Check: `.venv/bin/python scripts/check_release.py` (each released checkpoint loaded as a user would, compared with
   its training run; probabilities sum to 1), then `examples/quickstart.py` from a fresh clone.
4. Commit and push (no co-author trailers).
5. Hugging Face, from the clean commit:
   ```bash
   huggingface-cli login                                        # token with write access to the organization
   .venv/bin/python scripts/publish_hf.py --org <org> --dry-run  # list what would be uploaded
   .venv/bin/python scripts/publish_hf.py --org <org>            # private repos <org>/<checkpoint>
   ```
   Then `load_decider("<org>/opendecider-9b")` works anywhere (`opendecider.hub` downloads the repo and its pinned
   backbone). Make a repo public from its Hugging Face settings when ready.
