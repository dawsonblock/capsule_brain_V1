"""Report CLI for AutoLearn v0.3.1 (Section 24)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qualification.autolearn_v031.report import generate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate qualification report.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--active-policy-id", default="")
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    report = generate_report(artifacts, active_policy_id=args.active_policy_id)

    output_path = artifacts / "AUTOLEARN_V0_3_1_REPORT.md"
    output_path.write_text(report)
    print(f"Report written to: {output_path}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
