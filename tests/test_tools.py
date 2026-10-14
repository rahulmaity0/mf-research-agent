import pytest
from conftest import FakeSource, scorecard

from mfagent.client import FundNotFound
from mfagent.tools import build_tools, clean


def tools_for(source):
    return {t.name: t for t in build_tools(source)}


def test_clean_turns_fractions_into_percentages():
    out = clean({"ret_1y": 0.070455, "vol_1y": 0.1296})

    assert out == {"ret_1y_pct": 7.05, "vol_1y_pct": 12.96}


def test_clean_drops_missing_values_and_ranks_of_missing_metrics():
    out = clean(scorecard(1))

    assert "cagr_3y_pct" not in out
    # cagr_3y is null for every fund, so the pipeline ranks them all 1; that rank must not reach the model
    assert "rank_cagr_3y" not in out
    assert out["rank_ret_1y"] == 1


def test_list_categories_filters_by_keyword_case_insensitively():
    result = tools_for(FakeSource())["list_categories"].invoke({"keyword": "LARGE CAP"})

    assert [c["category"] for c in result] == ["Equity Scheme - Large Cap Fund", "Equity Schemes - Large Cap Fund"]


def test_search_funds_ranks_returns_highest_first():
    source = FakeSource([scorecard(1, ret_1y=0.05), scorecard(2, ret_1y=0.20), scorecard(3, ret_1y=None)])

    result = tools_for(source)["search_funds"].invoke({"category": "Equity Scheme - Large Cap Fund"})

    assert [r["scheme_code"] for r in result] == [2, 1, 3]


def test_search_funds_ranks_volatility_lowest_first():
    source = FakeSource([scorecard(1, vol_1y=0.20), scorecard(2, vol_1y=0.10), scorecard(3, vol_1y=0.15)])

    result = tools_for(source)["search_funds"].invoke(
        {"category": "Equity Scheme - Large Cap Fund", "metric": "vol_1y"})

    assert [r["scheme_code"] for r in result] == [2, 3, 1]


def test_search_funds_caps_limit_at_20():
    source = FakeSource([scorecard(i) for i in range(30)])

    result = tools_for(source)["search_funds"].invoke({"category": "Equity Scheme - Large Cap Fund", "limit": 500})

    assert len(result) == 20


def test_search_funds_in_unknown_category_tells_the_model_to_check_names():
    result = tools_for(FakeSource())["search_funds"].invoke({"category": "Large Cap"})

    assert "list_categories" in result[0]["error"]


def test_search_funds_rejects_metric_outside_allowed_list():
    with pytest.raises(Exception, match="metric"):
        tools_for(FakeSource())["search_funds"].invoke({"category": "x", "metric": "nav"})


def test_get_fund_details_for_unknown_fund_raises_not_found():
    with pytest.raises(FundNotFound):
        tools_for(FakeSource())["get_fund_details"].invoke({"scheme_code": 999})
