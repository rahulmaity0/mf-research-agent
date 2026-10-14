"""The tools the LLM can call.

Every number the agent reports comes out of these functions. The model decides
which tool to call and words the answer, but it never computes a return or a
ranking itself: those come from the pipeline, and anything that needs sorting or
formatting is done here in Python.
"""

from typing import Literal

from langchain_core.tools import BaseTool, tool

from mfagent.client import FundSource

# Fields the pipeline returns as fractions (0.07 = 7%)
PERCENT_FIELDS = {
    "ret_1m", "ret_1y", "cagr_3y", "cagr_5y", "vol_1y", "max_drawdown_1y",
    "category_median_ret_1y", "category_median_cagr_3y",
    "ret_1y_vs_category", "ret_1y_vs_benchmark", "cagr_3y_vs_benchmark", "cagr_5y_vs_benchmark",
}

# A rank is only meaningful if the metric it ranks exists. When every fund's
# cagr_3y is null, the pipeline's rank() gives all of them rank 1.
RANK_OF = {"rank_ret_1y": "ret_1y", "rank_cagr_3y": "cagr_3y", "rank_sharpe_1y": "sharpe_1y"}

Metric = Literal["ret_1y", "cagr_3y", "cagr_5y", "sharpe_1y", "vol_1y"]
LOWER_IS_BETTER = {"vol_1y"}

# Large enough to hold every fund in the biggest category we search
CATEGORY_FETCH_LIMIT = 200


def clean(row: dict) -> dict:
    """Drop missing values and ranks of missing metrics; turn fractions into percentages."""
    out = {}
    for key, value in row.items():
        if value is None:
            continue
        if key in RANK_OF and row.get(RANK_OF[key]) is None:
            continue
        if key in PERCENT_FIELDS:
            out[key + "_pct"] = round(value * 100, 2)
        elif isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def build_tools(source: FundSource) -> list[BaseTool]:
    @tool
    def list_categories(keyword: str | None = None) -> list[dict]:
        """List fund categories with how many funds each has. Category names are messy
        (e.g. both 'Equity Scheme - Large Cap Fund' and 'Equity Schemes - Large Cap Fund'
        exist), so call this first to find the exact names before searching.

        Args:
            keyword: optional case-insensitive filter, e.g. 'large cap' or 'liquid'
        """
        categories = source.list_categories()
        if keyword:
            categories = [c for c in categories if keyword.lower() in c["category"].lower()]
        return [{"category": c["category"], "plan": c["plan"], "funds": c["funds"]} for c in categories]

    @tool
    def search_funds(category: str, metric: Metric = "ret_1y",
                     plan: Literal["Direct", "Regular"] = "Direct", limit: int = 5) -> list[dict]:
        """Best funds in one category by a metric. 'Best' means highest for returns and
        Sharpe ratio, and lowest for volatility (vol_1y). Funds missing the metric go last.

        Args:
            category: exact category name from list_categories
            metric: ret_1y (1-year return), cagr_3y, cagr_5y, sharpe_1y (risk-adjusted), vol_1y (volatility)
            plan: Direct (lower fees) or Regular
            limit: how many funds to return, 1-20
        """
        limit = max(1, min(limit, 20))
        rows = source.search_funds(category, plan=plan, limit=CATEGORY_FETCH_LIMIT)
        if not rows:
            return [{"error": f"no {plan} funds in category {category!r}; check the name with list_categories"}]
        sign = 1 if metric in LOWER_IS_BETTER else -1
        rows.sort(key=lambda r: (r[metric] is None, sign * (r[metric] or 0), r["scheme_code"]))
        return [clean(r) for r in rows[:limit]]

    @tool
    def get_fund_details(scheme_code: int) -> dict:
        """Full latest scorecard for one fund: returns, volatility, drawdown, Sharpe,
        comparison with its category median and the Nifty 50, and its rank in the category.

        Args:
            scheme_code: AMFI scheme code, from search_funds results
        """
        fund = source.get_fund(scheme_code)
        registry = fund.get("registry") or {}
        details = clean(fund["scorecard"])
        details["name_changes"] = registry.get("name_changes", 0)
        details["first_nav_date"] = registry.get("first_nav_date")
        return details

    @tool
    def get_fund_history(scheme_code: int, months: int = 12) -> list[dict]:
        """Month-end metrics for one fund, newest month first. Use it to see how a fund's
        return or volatility moved over time.

        Args:
            scheme_code: AMFI scheme code
            months: how many months back, 1-12
        """
        months = max(1, min(months, 12))
        return [clean(row) for row in source.get_history(scheme_code, months)]

    return [list_categories, search_funds, get_fund_details, get_fund_history]
