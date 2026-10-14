"""Record a snapshot of the pipeline API into data/demo/ for demo mode and tests.

Needs mf-analytics-pipeline's API running (docker compose up -d postgres mongo redis api).

    python scripts/record_demo_data.py --base-url http://localhost:8000
"""

import argparse
import json
from pathlib import Path

from mfagent.client import FundApiClient

# Common categories only, to keep the snapshot small. AMFI renamed "Equity Scheme"
# to "Equity Schemes" partway through, so both spellings are kept.
KEEP = (
    "Large Cap Fund", "Mid Cap Fund", "Small Cap Fund", "Large & Mid Cap Fund", "Flexi Cap Fund",
    "Multi Cap Fund", "Focused Fund", "Value Fund", "ELSS", "Liquid Fund",
    "Aggressive Hybrid Fund", "Balanced Advantage",
)
HISTORY_MONTHS = 12
OUT = Path(__file__).resolve().parent.parent / "data" / "demo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    api = FundApiClient(args.base_url, timeout=60)

    categories = [c for c in api.list_categories() if any(k in c["category"] for k in KEEP)]
    funds, histories = {}, {}
    for cat in {c["category"] for c in categories}:
        for row in api.search_funds(cat, limit=200):
            code = row["scheme_code"]
            funds[str(code)] = api.get_fund(code)
            histories[str(code)] = api.get_history(code, HISTORY_MONTHS)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("categories", categories), ("funds", funds), ("histories", histories)):
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"recorded {len(categories)} categories, {len(funds)} funds into {OUT}")


if __name__ == "__main__":
    main()
