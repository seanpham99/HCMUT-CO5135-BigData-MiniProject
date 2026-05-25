from dags.etl_modules.fetcher import fetch_stock_price, get_active_vn_stock_tickers


class PriceProviderAdapter:
    def list_assets(self) -> list[dict[str, str]]:
        return get_active_vn_stock_tickers(raise_on_fallback=True)

    def fetch_prices(
        self,
        symbol: str,
        asset_id: str,
        start_date: str,
        end_date: str,
    ):
        return fetch_stock_price(symbol, asset_id, start_date, end_date)
