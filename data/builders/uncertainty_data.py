"""Uncertainty-shaping TRAINING data from the synthetic suite (idea documented by Kev; targets calibration):

  unknowable : a question paired with an UNRELATED state from another family (the deciding evidence is absent);
               target = uniform distribution over the options ("soft" field), trained with the proper-scoring loss.
  buried     : the original state embedded in 2-5k characters of unrelated records; label unchanged.

    python data/builders/uncertainty_data.py     # -> data/synthetic/uncertainty_train.jsonl
"""
import json
import random


def main(n_unknowable=3000, n_buried=3000, seed=0):
    rng = random.Random(seed)
    recs = [json.loads(l) for l in open("data/synthetic/train.jsonl")]
    by_fam = {}
    for r in recs:
        by_fam.setdefault(r["family"], []).append(r)
    fams = sorted(by_fam)
    out = []
    for i in range(n_unknowable):
        r = rng.choice(recs)
        donor = rng.choice(by_fam[rng.choice([f for f in fams if f != r["family"]])])
        qs = []
        for q in rng.sample(r["questions"], min(3, len(r["questions"]))):
            K = len(q["options"])
            qs.append(dict(q, soft=[1.0 / K] * K))
        out.append({"id": f"unknowable-{i}", "family": f"unknowable_{r['family']}", "state": donor["state"], "questions": qs})
    for i in range(n_buried):
        r = rng.choice(recs)
        filler, target = [], rng.randint(2000, 5000)
        while sum(len(json.dumps(x)) for x in filler) < target:
            filler.append(rng.choice(by_fam[rng.choice([f for f in fams if f != r["family"]])])["state"])
        cut = rng.randint(0, len(filler))
        state = {"background": filler[:cut], "record": r["state"], "appendix": filler[cut:]}
        out.append({"id": f"buried-{i}", "family": f"buried_{r['family']}", "state": state, "questions": r["questions"]})
    rng.shuffle(out)
    with open("data/synthetic/uncertainty_train.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"uncertainty_train: {len(out)} records ({n_unknowable} unknowable, {n_buried} buried)")


if __name__ == "__main__":
    main()
