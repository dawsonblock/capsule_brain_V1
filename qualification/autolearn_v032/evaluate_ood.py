"""Evaluate candidate on the out-of-distribution split (Section 10)."""
import sys
from .evaluate import main_for_split

def main() -> int:
    return main_for_split("ood")

if __name__ == "__main__":
    sys.exit(main())
