"""Command line: ``python -m parkwise build | serve | evaluate``."""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="parkwise", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="simulate data, train models, evaluate, write artifacts/")
    b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("serve", help="run the API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("evaluate", help="print the headline numbers from artifacts/report.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "build":
        from .pipeline import build
        build(seed=args.seed)
        return 0
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("parkwise.api:app", host=args.host, port=args.port, log_level="info")
        return 0
    if args.cmd == "evaluate":
        from .config import ARTIFACTS
        path = ARTIFACTS / "report.json"
        if not path.exists():
            print("No report found. Run: python -m parkwise build", file=sys.stderr)
            return 1
        r = json.loads(path.read_text())
        f, p = r["forecast"], r["policy"]
        print(f"Forecast WMAPE  GBM {f['point']['gbm']['wmape']:.1%}  | seasonal-naive {f['point']['seasonal_naive']['wmape']:.1%}")
        print(f"80% interval coverage {f['interval']['coverage']:.1%}")
        dm = p["policies"]["dynamic_model"]
        print(f"Pricing replay  revenue {dm['revenue_uplift_vs_status_quo']:+.1%} "
              f"(95% CI {dm['uplift_ci95'][0]:+.1%}..{dm['uplift_ci95'][1]:+.1%}) vs static schedule")
        print(f"Ticket classifier macro-F1 {r['tickets']['model']['macro_f1']:.3f} "
              f"(keyword baseline {r['tickets']['keyword_baseline']['macro_f1']:.3f})")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
