from mfagent.client import FundNotFound


def scorecard(code, ret_1y=0.1, vol_1y=0.15, category="Equity Scheme - Large Cap Fund", plan="Direct"):
    return {
        "scheme_code": code, "scheme_name": f"Fund {code}", "category": category, "plan": plan,
        "as_of_date": "2026-10-07", "nav": 100.123456,
        "ret_1y": ret_1y, "vol_1y": vol_1y, "sharpe_1y": 0.5,
        "cagr_3y": None, "cagr_5y": None,
        "rank_ret_1y": 1, "rank_cagr_3y": 1, "rank_sharpe_1y": 1,
    }


class FakeSource:
    """In-memory FundSource for tests."""

    def __init__(self, funds: list[dict] | None = None):
        self.funds = {f["scheme_code"]: f for f in funds or []}

    def list_categories(self):
        return [
            {"category": "Equity Scheme - Large Cap Fund", "plan": "Direct", "funds": 24},
            {"category": "Equity Schemes - Large Cap Fund", "plan": "Direct", "funds": 11},
            {"category": "Debt Scheme - Liquid Fund", "plan": "Direct", "funds": 24},
        ]

    def search_funds(self, category, plan=None, sort_by="ret_1y", limit=10):
        rows = [f for f in self.funds.values() if f["category"] == category and (plan is None or f["plan"] == plan)]
        return rows[:limit]

    def get_fund(self, scheme_code):
        if scheme_code not in self.funds:
            raise FundNotFound(f"scheme {scheme_code} not found")
        return {"scorecard": self.funds[scheme_code], "registry": {"name_changes": 0}}

    def get_history(self, scheme_code, months=12):
        if scheme_code not in self.funds:
            raise FundNotFound(f"no metrics for scheme {scheme_code}")
        return [{"month": "2026-10-01", "ret_1y": 0.1, "cagr_3y": None}][:months]
