"""Command-line entry point: ``yand-mvsk serve`` and ``yand-mvsk optimize``."""

from __future__ import annotations

import argparse
import json
import sys


def _serve(args) -> int:
    import uvicorn

    print(f"\n  YAND-MVSK Tail-Risk Studio → http://{args.host}:{args.port}\n")
    uvicorn.run(
        "yand_mvsk.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=["src"] if args.reload else None,
        log_level="info",
    )
    return 0


def _optimize(args) -> int:
    from .api.service import optimize_portfolio

    answers = json.loads(args.answers) if args.answers else None
    out = optimize_portfolio(
        tickers=args.tickers,
        gamma=args.gamma,
        answers=answers,
        max_weight=args.max_weight,
        horizon=args.horizon,
        offline=args.offline,
        lookback=args.lookback,
    )
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        print()
        return 0

    p, r = out["performance"], out["risk"]
    print(f"\n  Gamma (risk aversion): {out['gamma']}"
          + (f"  ·  {out['gamma_breakdown']['profile']}" if out["gamma_breakdown"] else ""))
    print(f"  Data source: {out['data_source']}  ·  {out['n_observations']} days "
          f"({out['date_range'][0]} → {out['date_range'][1]})\n")
    print("  Optimal weights")
    print("  " + "-" * 46)
    for row in out["weights"]:
        print(f"    {row['symbol']:12s} {row['weight']*100:6.2f}%   {row['market_label']:16s} {row['name']}")
    print()
    print(f"  Ann. return {p['ann_return']*100:+.1f}%   Ann. vol {p['ann_volatility']*100:.1f}%   "
          f"Sharpe {p['sharpe']:.2f}   Sortino {p['sortino']:.2f}")
    print(f"  Skewness {p['skewness']:+.2f}   Excess kurtosis {p['excess_kurtosis']:+.2f}   "
          f"95% CVaR {p['cvar_95']*100:.2f}%   MaxDD {p['max_drawdown']*100:.1f}%")
    print(f"\n  Tail-Risk Guard: {r['score']}/100 ({r['level']}) — {r['headline']}")
    for f in r["findings"]:
        print(f"    • {f}")
    if out.get("recommendation"):
        print(f"\n  → {out['recommendation']['how']}")
    print(f"\n  Solved in {p['solver']['seconds']*1000:.0f} ms "
          f"({p['solver']['iterations']} iterations).\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="yand-mvsk",
        description="Higher-moment (MVSK) portfolio optimization with a tail-risk guard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("serve", help="Run the web studio (API + frontend).")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--reload", action="store_true", help="Auto-reload on source changes (dev).")
    ps.set_defaults(func=_serve)

    po = sub.add_parser("optimize", help="Optimize a basket from the command line.")
    po.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL 0700.HK 600519.SS")
    po.add_argument("--gamma", type=float, default=None, help="Risk aversion (overrides --answers).")
    po.add_argument("--answers", default=None, help='VBQ-5 answers as JSON, e.g. \'{"Q1":"B","Q2":"B","Q3":"A","Q4":"A","Q5":"A"}\'')
    po.add_argument("--max-weight", type=float, default=0.35, dest="max_weight")
    po.add_argument("--horizon", type=int, default=21)
    po.add_argument("--lookback", default="3y")
    po.add_argument("--offline", action="store_true", help="Use deterministic synthetic data.")
    po.add_argument("--json", action="store_true", help="Emit the full result as JSON.")
    po.set_defaults(func=_optimize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
