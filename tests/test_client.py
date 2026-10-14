import json

import httpx
import pytest

from mfagent.client import DemoFundClient, FundApiClient, FundApiError, FundNotFound


def live_client(handler) -> FundApiClient:
    return FundApiClient("http://pipeline", transport=httpx.MockTransport(handler))


# --- FundApiClient ---------------------------------------------------------

def test_search_funds_sends_filters_and_skips_missing_plan():
    seen = {}

    def handler(request: httpx.Request):
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[{"scheme_code": 1}])

    rows = live_client(handler).search_funds("Equity Scheme - Large Cap Fund", sort_by="sharpe_1y", limit=5)

    assert rows == [{"scheme_code": 1}]
    assert seen["path"] == "/funds"
    assert seen["params"] == {"category": "Equity Scheme - Large Cap Fund", "sort_by": "sharpe_1y", "limit": "5"}


def test_404_becomes_fund_not_found():
    client = live_client(lambda r: httpx.Response(404, json={"detail": "scheme 999 not found"}))

    with pytest.raises(FundNotFound, match="scheme 999 not found"):
        client.get_fund(999)


def test_server_error_becomes_fund_api_error():
    client = live_client(lambda r: httpx.Response(500, text="boom"))

    with pytest.raises(FundApiError, match="500"):
        client.list_categories()


def test_unreachable_api_becomes_fund_api_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(FundApiError, match="unreachable"):
        live_client(handler).list_categories()


# --- DemoFundClient --------------------------------------------------------

def scorecard(code, category="Equity Scheme - Large Cap Fund", plan="Direct", ret_1y=0.1):
    return {"scheme_code": code, "category": category, "plan": plan, "ret_1y": ret_1y,
            "sharpe_1y": None, "cagr_3y": None, "cagr_5y": None, "vol_1y": 0.1}


@pytest.fixture
def demo(tmp_path) -> DemoFundClient:
    funds = {
        "1": {"scorecard": scorecard(1, ret_1y=0.05)},
        "2": {"scorecard": scorecard(2, ret_1y=None)},
        "3": {"scorecard": scorecard(3, ret_1y=0.20)},
        "4": {"scorecard": scorecard(4, plan="Regular", ret_1y=0.30)},
        "5": {"scorecard": scorecard(5, category="Equity Scheme - Mid Cap Fund", ret_1y=0.50)},
    }
    histories = {"1": [{"month": "2026-10-01"}, {"month": "2026-09-01"}, {"month": "2026-08-01"}]}
    (tmp_path / "categories.json").write_text("[]")
    (tmp_path / "funds.json").write_text(json.dumps(funds))
    (tmp_path / "histories.json").write_text(json.dumps(histories))
    return DemoFundClient(tmp_path)


def test_demo_search_filters_by_category_and_plan_and_sorts_nulls_last(demo):
    rows = demo.search_funds("Equity Scheme - Large Cap Fund", plan="Direct")

    assert [r["scheme_code"] for r in rows] == [3, 1, 2]


def test_demo_search_applies_limit(demo):
    assert len(demo.search_funds("Equity Scheme - Large Cap Fund", limit=1)) == 1


def test_demo_search_rejects_unknown_sort_column(demo):
    with pytest.raises(FundApiError, match="sort_by"):
        demo.search_funds("Equity Scheme - Large Cap Fund", sort_by="nav")


def test_demo_unknown_fund_is_not_found(demo):
    with pytest.raises(FundNotFound):
        demo.get_fund(999)


def test_demo_history_returns_newest_months_first(demo):
    assert demo.get_history(1, months=2) == [{"month": "2026-10-01"}, {"month": "2026-09-01"}]


def test_demo_history_for_fund_without_metrics_is_not_found(demo):
    with pytest.raises(FundNotFound):
        demo.get_history(3)
