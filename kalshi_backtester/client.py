"""Kalshi REST API client with RSA-PSS authentication."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Iterator

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
PROD_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class KalshiClient:
    """Thin wrapper around the Kalshi v2 REST API.

    Authentication uses RSA-PSS signing per Kalshi's spec:
      KALSHI-ACCESS-KEY        -> your API key ID
      KALSHI-ACCESS-TIMESTAMP  -> Unix milliseconds as a string
      KALSHI-ACCESS-SIGNATURE  -> base64(RSA-PSS-SHA256(timestamp + METHOD + path))
    The path used for signing must NOT include query parameters.
    """

    def __init__(
        self,
        api_key_id: str,
        private_key: str | bytes | Path,
        env: str = "demo",
    ) -> None:
        if env == "prod":
            self.base_url = PROD_BASE_URL
        else:
            self.base_url = DEMO_BASE_URL

        self.api_key_id = api_key_id
        self._private_key = self._load_key(private_key)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_key(key: str | bytes | Path):
        if isinstance(key, Path):
            raw = key.read_bytes()
        elif isinstance(key, str):
            raw = key.encode()
        else:
            raw = key
        return serialization.load_pem_private_key(raw, password=None)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        message = (timestamp_ms + method.upper() + path).encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
        }

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self.base_url + path
        headers = self._auth_headers("GET", path)
        resp = self._session.get(url, headers=headers, params=params, timeout=30)
        if not resp.ok:
            raise KalshiAPIError(resp.status_code, resp.text[:300])
        return resp.json()

    # ------------------------------------------------------------------
    # Market endpoints
    # ------------------------------------------------------------------

    def get_markets_page(
        self,
        status: str = "finalized",
        limit: int = 1000,
        cursor: str | None = None,
    ) -> dict:
        """Fetch one page of markets.  Returns the raw API response dict."""
        params: dict = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params)

    def iter_markets(
        self,
        status: str = "finalized",
        page_size: int = 1000,
    ) -> Iterator[dict]:
        """Yield every market object for the given status, handling pagination."""
        cursor: str | None = None
        while True:
            resp = self.get_markets_page(status=status, limit=page_size, cursor=cursor)
            markets = resp.get("markets") or []
            yield from markets
            cursor = resp.get("cursor")
            if not cursor or not markets:
                break

    def get_market(self, ticker: str) -> dict:
        """Fetch a single market by ticker.  Returns the market object."""
        resp = self._get(f"/markets/{ticker}")
        return resp.get("market", resp)

    def get_market_trades(
        self,
        ticker: str,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> dict:
        """Fetch historical trades for a market."""
        params: dict = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get(f"/markets/{ticker}/trades", params)
