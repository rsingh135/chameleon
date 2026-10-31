"""Train the lightweight local fit policy from explicit development comparisons."""

import argparse
import json
from pathlib import Path

from engine.fit_policy import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.dataset.stat().st_size > 64 * 1024 * 1024:
        parser.error("Training dataset exceeds the bounded local byte budget")
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        artifact = train_model(dataset, args.out, epochs=args.epochs, seed=args.seed)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Fit policy training failed: {exc}\n")
    print(
        json.dumps(
            {
                "model": str(args.out),
                "model_digest": artifact["model_digest"],
                "training": artifact["training"],
                "metrics": artifact["metrics"],
                "quality_claim": artifact["quality_claim"],
                "vlm_fine_tuning": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
