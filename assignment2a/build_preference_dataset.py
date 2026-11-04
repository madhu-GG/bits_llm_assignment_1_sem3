"""Generate preference_dataset.jsonl from the Part A / Part B strategy outputs.

Run with the offline backend on CPU to produce a reproducible dataset, or pass
--backend adapter on a GPU runtime to build it from real adapter generations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import preference_data as pd  # noqa: E402
from a2_paths import DEFAULT_SFT_ADAPTER, PREFERENCE_DATASET, base_model_source, sft_adapter_path  # noqa: E402
from domain_lm import DomainLM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["offline", "adapter"], default="offline")
    parser.add_argument("--adapter", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--out", default=str(PREFERENCE_DATASET))
    parser.add_argument("--min-pairs", type=int, default=60)
    args = parser.parse_args()

    adapter = sft_adapter_path(args.adapter)
    lm = DomainLM(
        base_model_id=base_model_source(),
        adapter_path=adapter if args.backend == "adapter" else None,
        backend=args.backend,
        label=f"sft_{args.adapter}",
    )
    print("generator:", json.dumps(lm.describe(), indent=2))

    pairs, stats = pd.build_preference_pairs(lm, min_pairs=args.min_pairs)
    problems = pd.validate(pairs, min_pairs=args.min_pairs)
    path = pd.save_jsonl(pairs, args.out)

    print("\nstats:", json.dumps(stats, indent=2))
    print("written:", path)
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("validation: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
