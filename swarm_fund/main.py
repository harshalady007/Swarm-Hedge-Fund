"""Swarm signal desk for NSE large caps. Paper trading only; it never places orders.

    python main.py run [--offline] [--asof YYYY-MM-DD] [--db PATH]
    python main.py report [--db PATH]
    python main.py doctor [--symbol SYM] [--db PATH]
"""
import argparse
import sys

import config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="daily run after the NSE close")
    r.add_argument("--offline", action="store_true", help="synthetic data and rule-based agents; no network")
    r.add_argument("--asof", help="pretend today is this date (YYYY-MM-DD)")
    r.add_argument("--db", default=config.DEFAULT_DB)
    rep = sub.add_parser("report", help="paper portfolio record")
    rep.add_argument("--db", default=config.DEFAULT_DB)
    d = sub.add_parser("doctor", help="call every data source once and print a status table")
    d.add_argument("--symbol", default="RELIANCE")
    d.add_argument("--db", default=config.DEFAULT_DB)
    args = p.parse_args(argv)

    if args.cmd == "run":
        import pipeline
        try:
            pipeline.run(args.db, offline=args.offline, asof=args.asof)
        except pipeline.RunError as e:
            print(f"RUN ABORTED: {e}", file=sys.stderr)
            return 2
    elif args.cmd == "report":
        import db
        import journal
        print(journal.report(db.connect(args.db)))
    elif args.cmd == "doctor":
        import doctor
        return doctor.run(args.symbol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
