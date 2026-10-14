"""Clients for the mf-analytics-pipeline REST API.

Two implementations with the same methods:

- FundApiClient talks to a running pipeline over HTTP.
- DemoFundClient reads a snapshot recorded from that API (data/demo/), so the
  agent can run without the pipeline's Docker stack.

The agent only ever sees the FundSource interface, so it doesn't care which one
it gets.
"""

import json
from pathlib import Path
from typing import Protocol

import httpx

SORT_COLUMNS = {"ret_1y", "cagr_3y", "cagr_5y", "sharpe_1y", "vol_1y"}


class FundApiError(Exception):
    """The pipeline API could not be reached or returned something unexpected."""


class FundNotFound(FundApiError):
    """No fund (or no history) exists for the requested scheme code."""


class FundSource(Protocol):
    def list_categories(self) -> list[dict]: ...

    def search_funds(self, category: str, plan: str | None = None,
                     sort_by: str = "ret_1y", limit: int = 10) -> list[dict]: ...

    def get_fund(self, scheme_code: int) -> dict: ...

    def get_history(self, scheme_code: int, months: int = 12) -> list[dict]: ...


class FundApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0, transport: httpx.BaseTransport | None = None):
        # transport is only passed by tests, to fake the server
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _get(self, path: str, params: dict | None = None):
        try:
            response = self._http.get(path, params=params)
        except httpx.HTTPError as e:
            raise FundApiError(f"pipeline API unreachable: {e}") from e
        if response.status_code == 404:
            raise FundNotFound(response.json().get("detail", "not found"))
        if response.status_code != 200:
            raise FundApiError(f"pipeline API returned {response.status_code}: {response.text[:200]}")
        return response.json()

    def list_categories(self) -> list[dict]:
        return self._get("/categories")

    def search_funds(self, category: str, plan: str | None = None,
                     sort_by: str = "ret_1y", limit: int = 10) -> list[dict]:
        params = {"category": category, "sort_by": sort_by, "limit": limit}
        if plan:
            params["plan"] = plan
        return self._get("/funds", params)

    def get_fund(self, scheme_code: int) -> dict:
        return self._get(f"/funds/{scheme_code}")

    def get_history(self, scheme_code: int, months: int = 12) -> list[dict]:
        return self._get(f"/funds/{scheme_code}/history", {"months": months})


class DemoFundClient:
    """Answers from a recorded snapshot instead of a live API.

    Filtering and sorting mirror what the pipeline's API does, so the agent
    behaves the same in both modes.
    """

    def __init__(self, data_dir: Path):
        self._categories = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))
        self._funds = json.loads((data_dir / "funds.json").read_text(encoding="utf-8"))
        self._histories = json.loads((data_dir / "histories.json").read_text(encoding="utf-8"))

    def list_categories(self) -> list[dict]:
        return self._categories

    def search_funds(self, category: str, plan: str | None = None,
                     sort_by: str = "ret_1y", limit: int = 10) -> list[dict]:
        if sort_by not in SORT_COLUMNS:
            raise FundApiError(f"sort_by must be one of {sorted(SORT_COLUMNS)}")
        rows = [f["scorecard"] for f in self._funds.values()
                if f["scorecard"]["category"] == category and (plan is None or f["scorecard"]["plan"] == plan)]
        # Same as the API's "order by <col> desc nulls last, scheme_code"
        rows.sort(key=lambda r: (r[sort_by] is None, -(r[sort_by] or 0), r["scheme_code"]))
        return rows[:limit]

    def get_fund(self, scheme_code: int) -> dict:
        fund = self._funds.get(str(scheme_code))
        if fund is None:
            raise FundNotFound(f"scheme {scheme_code} not found")
        return fund

    def get_history(self, scheme_code: int, months: int = 12) -> list[dict]:
        history = self._histories.get(str(scheme_code))
        if not history:
            raise FundNotFound(f"no metrics for scheme {scheme_code}")
        # Stored newest month first, like the API returns it
        return history[:months]
