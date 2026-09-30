#!/usr/bin/env bash
# Recipe ablation ladder on the 2B (each arm = previous + one change), equal budget, targeted eval after each.
#   R0 base (x2b recipe) | R1 +question cache & full listing (cap 255, 64 train options) | R2 +schema-example data
#   R3 R2 with VeRA on rows only (state pass frozen -> cacheable, faster) | R4 R2 with grad_accum 4 at EQUAL data/compute
cd "$(dirname "$0")/.."
C=configs/train_2b.yaml
S="train.max_steps=600 train.eval_every=600 train.save_every=600"
Q="model.v3_question_cache=true model.v3_list_cap=255 train.max_train_options=64"
D="train.data=[data/synthetic/train.jsonl,data/public_train/train.jsonl,data/public_train/injection_train.jsonl,data/public_train/knowledge_train.jsonl,data/public_train/context_train.jsonl,data/synthetic/uncertainty_train.jsonl]"
OLD="train.data=[data/synthetic/train.jsonl,data/public_train/train.jsonl,data/public_train/injection_train.jsonl,data/public_train/knowledge_train.jsonl,data/public_train/context_train_v1.jsonl,data/synthetic/uncertainty_train.jsonl]"
P=data/public; X=data/public_context
arm() {  # name, overrides...
  local n=$1; shift
  if [ -f "runs/abl-$n/eval_banking77_8way.json" ]; then echo "$n done (already evaluated)"; return; fi
  if [ ! -f "runs/$n/model.pt" ]; then
    scripts/launch_train.sh "$n" "$C" "$@"      # launch_train already opens `--set`: overrides append to it
    sleep 5; while kill -0 "$(cat runs/$n.pid)" 2>/dev/null; do sleep 30; done
  fi
  [ -f "runs/$n/model.pt" ] || { echo "$n: no checkpoint"; return; }
  .venv/bin/python -m eval.evaluate --ckpt "runs/$n/model.pt" --save-preds --n-boot 500 --limit 300 --name "abl-$n" \
      --data $P/banking77_77way.jsonl $P/banking77_8way.jsonl $P/prompt_injections.jsonl $P/openbookqa.jsonl \
      $P/commonsenseqa.jsonl $X/batchdep_schema.jsonl $X/banking77_8way_icl4_schema.jsonl > "runs/abl-$n.log" 2>&1
  echo "$n done"
}
arm r0 $S $OLD
arm r1 $S $OLD $Q
arm r2 $S $D $Q
arm r3 $S $D $Q model.v3_vera_scope=rows_top
arm r4 train.max_steps=150 train.eval_every=150 train.save_every=150 train.grad_accum=4 $D $Q
