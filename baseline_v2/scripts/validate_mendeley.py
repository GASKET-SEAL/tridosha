"""Reproduce every data result quoted in the project report.   python scripts/validate_mendeley.py"""
import sys, json; sys.path.insert(0, ".")
from dosha.prakriti import cross_cohort, transparent_oos, check_expert_scoring
print("1) ML (softmax) cross-cohort replication, 105 shared features"); print(json.dumps(cross_cohort(), indent=1))
print("\n2) Transparent vote-count, rules learned on one cohort, scored on the other (honest out-of-sample)"); print(json.dumps(transparent_oos(), indent=1))
r = check_expert_scoring(); print(f"\n3) In-sample vote-count over 10 hand-mapped traits: {r['accuracy']:.3f} (n={r['n']}; optimistic, traits chosen on same data)")
print(r["per_answer"].round(2).to_string(index=False))
