#!/usr/bin/env python
# coding: utf-8

"""
Market data layer — Yahoo Finance edition.

This module was migrated from Financial Modeling Prep (FMP) to Yahoo Finance
(via the `yfinance` package) so that the equity research pipeline works for
global markets, including Indian listings (e.g. ADANIENT.NS, RELIANCE.NS),
without requiring a paid FMP subscription.

IMPORTANT: every public function keeps its original signature (including the now
unused `api_key` argument) and, crucially, returns data using the SAME column /
field names that the downstream processor expects (the historical FMP naming
convention such as `revenue`, `costOfRevenue`, `ebitda`, `eps`,
`priceEarningsRatio`, `returnOnEquity`, etc.). This keeps the rest of the
pipeline unchanged.
"""

import datetime
import os

import numpy as np
import pandas as pd
import yfinance as yf

# Assuming common_utils.py is in the same parent directory (src/modules)
from .common_utils import get_api_key, load_config  # noqa: F401  (kept for API compat)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥",
    "CNY": "¥", "HKD": "HK$", "AUD": "A$", "CAD": "C$", "SGD": "S$",
    "CHF": "CHF ", "KRW": "₩", "BRL": "R$", "ZAR": "R", "AED": "AED ",
}

_EXCHANGE_NAMES = {
    "NSI": "NSE", "BSE": "BSE", "NMS": "NASDAQ", "NGM": "NASDAQ",
    "NYQ": "NYSE", "PCX": "NYSE", "ASE": "AMEX", "LSE": "LSE",
}

_ticker_cache: dict = {}


def _get_ticker(symbol: str) -> yf.Ticker:
    """Return a cached yfinance Ticker object."""
    if symbol not in _ticker_cache:
        _ticker_cache[symbol] = yf.Ticker(symbol)
    return _ticker_cache[symbol]


def _get_info(symbol: str) -> dict:
    """Safely fetch the yfinance .info dict."""
    try:
        info = _get_ticker(symbol).info or {}
        return info
    except Exception as e:
        print(f"Warning: could not fetch yfinance info for {symbol}: {e}")
        return {}


def currency_symbol_for(symbol: str) -> str:
    """Return the currency symbol (e.g. ₹ for an .NS ticker)."""
    info = _get_info(symbol)
    return _CURRENCY_SYMBOLS.get(info.get("currency", "USD"), "")


def _first_row(df: pd.DataFrame, *candidates):
    """Return the first matching row (as a Series indexed by date) from a
    transposed-yfinance statement, trying several possible line-item names."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _statement_to_fmp_df(raw: pd.DataFrame, field_map: dict) -> pd.DataFrame | None:
    """Convert a raw yfinance financial statement (rows = line items, columns =
    period end dates) into a row-per-year DataFrame using FMP field names.

    Args:
        raw: yfinance statement DataFrame (e.g. ticker.income_stmt)
        field_map: {fmp_field_name: [possible yfinance line item names]}
    """
    if raw is None or raw.empty:
        return None

    # Transpose -> index becomes the period-end dates, columns become line items
    t = raw.T
    records = []
    for period_end, row in t.iterrows():
        try:
            date = pd.to_datetime(period_end)
        except Exception:
            continue
        rec = {"date": date, "year": date.year}
        for fmp_name, yf_names in field_map.items():
            val = None
            for yf_name in yf_names:
                if yf_name in row.index and pd.notna(row[yf_name]):
                    val = float(row[yf_name])
                    break
            rec[fmp_name] = val
        records.append(rec)

    if not records:
        return None

    df = pd.DataFrame(records).sort_values("date", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Price / volume history
# --------------------------------------------------------------------------- #

def fetch_yfinance_volume(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetches historical trading volume data using yfinance."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            print(f"No data returned from yfinance for {ticker} between {start_date} and {end_date}")
            return None
        # yfinance may return a MultiIndex column frame when one ticker is passed
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        stock_data = stock_data[["Volume"]]
        stock_data.reset_index(inplace=True)
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])
        return stock_data
    except Exception as e:
        print(f"Error fetching yfinance volume for {ticker}: {e}")
        return None


def _history(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        hist = _get_ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching price history for {ticker}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Enterprise value
# --------------------------------------------------------------------------- #

def fetch_fmp_enterprise_value(ticker: str, api_key: str = None, limit: int = 2000) -> pd.DataFrame | None:
    """Returns current enterprise value as a single-row DataFrame.

    Yahoo Finance does not expose a historical EV time series, so we report the
    latest enterprise value from `.info`. Columns match the original FMP shape:
    ['date', 'enterpriseValue'].
    """
    info = _get_info(ticker)
    ev = info.get("enterpriseValue")
    if ev is None:
        print(f"No EV data available from Yahoo Finance for {ticker}.")
        return None
    df = pd.DataFrame(
        [{"date": pd.to_datetime(datetime.date.today()), "enterpriseValue": float(ev)}]
    )
    return df


# --------------------------------------------------------------------------- #
# Financial statements
# --------------------------------------------------------------------------- #

_INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "costOfRevenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "grossProfit": ["Gross Profit"],
    "operatingExpenses": ["Operating Expense"],
    "sellingGeneralAndAdministrativeExpenses": [
        "Selling General And Administration",
        "Selling General And Administrative Expense",
    ],
    "researchAndDevelopmentExpenses": ["Research And Development"],
    "operatingIncome": ["Operating Income", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "depreciationAndAmortization": [
        "Reconciled Depreciation",
        "Depreciation And Amortization In Income Statement",
        "Depreciation Amortization Depletion Income Statement",
    ],
    "interestExpense": ["Interest Expense", "Interest Expense Non Operating"],
    "incomeBeforeTax": ["Pretax Income"],
    "incomeTaxExpense": ["Tax Provision"],
    "netIncome": ["Net Income", "Net Income Common Stockholders"],
    "eps": ["Basic EPS"],
    "epsdiluted": ["Diluted EPS"],
    "weightedAverageShsOut": ["Basic Average Shares"],
    "weightedAverageShsOutDil": ["Diluted Average Shares"],
}


def _augment_income(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in EBITDA / EPS when Yahoo omits them, using available components."""
    for idx, row in df.iterrows():
        # Derive EBITDA if missing: Operating Income + D&A
        if (row.get("ebitda") is None) and row.get("operatingIncome") is not None:
            da = row.get("depreciationAndAmortization") or 0
            df.at[idx, "ebitda"] = row["operatingIncome"] + da
        # Derive EBITDA from pretax + interest + D&A as a secondary fallback
        if df.at[idx, "ebitda"] is None and row.get("incomeBeforeTax") is not None:
            interest = row.get("interestExpense") or 0
            da = row.get("depreciationAndAmortization") or 0
            df.at[idx, "ebitda"] = row["incomeBeforeTax"] + interest + da
        # Derive diluted EPS fallback to basic
        if df.at[idx, "epsdiluted"] is None and row.get("eps") is not None:
            df.at[idx, "epsdiluted"] = row["eps"]
        if df.at[idx, "eps"] is None and row.get("epsdiluted") is not None:
            df.at[idx, "eps"] = row["epsdiluted"]
    return df


def get_fmp_income_statement(ticker: str, api_key: str = None, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """Fetches income statement data from Yahoo Finance (FMP-compatible columns)."""
    try:
        tk = _get_ticker(ticker)
        raw = tk.quarterly_income_stmt if period == "quarter" else tk.income_stmt
        df = _statement_to_fmp_df(raw, _INCOME_MAP)
        if df is None:
            print(f"No income statement data from Yahoo Finance for {ticker}.")
            return None
        df = _augment_income(df)
        return df.head(limit)
    except Exception as e:
        print(f"Error fetching income statement for {ticker}: {e}")
        return None


_BALANCE_MAP = {
    "totalAssets": ["Total Assets"],
    "totalCurrentAssets": ["Current Assets"],
    "totalLiabilities": ["Total Liabilities Net Minority Interest"],
    "totalCurrentLiabilities": ["Current Liabilities"],
    "totalStockholdersEquity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "totalDebt": ["Total Debt"],
    "longTermDebt": ["Long Term Debt"],
    "shortTermDebt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "cashAndCashEquivalents": ["Cash And Cash Equivalents"],
    "cashAndShortTermInvestments": ["Cash Cash Equivalents And Short Term Investments"],
    "netDebt": ["Net Debt"],
    "commonStockSharesOutstanding": ["Ordinary Shares Number", "Share Issued"],
    "retainedEarnings": ["Retained Earnings"],
}


def get_fmp_balance_sheet(ticker: str, api_key: str = None, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """Fetches balance sheet data from Yahoo Finance (FMP-compatible columns)."""
    try:
        tk = _get_ticker(ticker)
        raw = tk.quarterly_balance_sheet if period == "quarter" else tk.balance_sheet
        df = _statement_to_fmp_df(raw, _BALANCE_MAP)
        if df is None:
            print(f"No balance sheet data from Yahoo Finance for {ticker}.")
            return None
        return df.head(limit)
    except Exception as e:
        print(f"Error fetching balance sheet for {ticker}: {e}")
        return None


_CASHFLOW_MAP = {
    "operatingCashFlow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "netCashProvidedByOperatingActivities": ["Operating Cash Flow"],
    "capitalExpenditure": ["Capital Expenditure"],
    "freeCashFlow": ["Free Cash Flow"],
    "netIncome": ["Net Income From Continuing Operations", "Net Income"],
    "depreciationAndAmortization": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
    "dividendsPaid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "stockBasedCompensation": ["Stock Based Compensation"],
}


def get_fmp_cash_flow_statement(ticker: str, api_key: str = None, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """Fetches cash flow statement data from Yahoo Finance (FMP-compatible columns)."""
    try:
        tk = _get_ticker(ticker)
        raw = tk.quarterly_cashflow if period == "quarter" else tk.cashflow
        df = _statement_to_fmp_df(raw, _CASHFLOW_MAP)
        if df is None:
            print(f"No cash flow data from Yahoo Finance for {ticker}.")
            return None
        return df.head(limit)
    except Exception as e:
        print(f"Error fetching cash flow for {ticker}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Ratios & key metrics (computed from statements + price history)
# --------------------------------------------------------------------------- #

def _year_end_prices(ticker: str, years: list[int]) -> dict:
    """Return {year: close price near Dec/fiscal year end} using daily history."""
    out = {}
    if not years:
        return out
    span = max(years) - min(years) + 2
    hist = _history(ticker, period=f"{max(span, 2)}y")
    if hist is None or hist.empty:
        return out
    closes = hist["Close"].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    for y in years:
        target = pd.Timestamp(year=y, month=12, day=31)
        sub = closes[closes.index <= target]
        if not sub.empty:
            out[y] = float(sub.iloc[-1])
    return out


def get_fmp_ratios_and_key_metrics(ticker: str, api_key: str = None, period: str = "annual", limit: int = 5) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Builds per-year financial ratios and key metrics from Yahoo Finance data.

    Returns (ratios_df, key_metrics_df) with FMP-compatible column names:
      ratios_df:      year, date, priceEarningsRatio, priceToBookRatio,
                      returnOnEquity, debtEquityRatio
      key_metrics_df: year, date, peRatio, pbRatio, enterpriseValueOverEBITDA
    """
    try:
        income = get_fmp_income_statement(ticker, period=period, limit=limit)
        balance = get_fmp_balance_sheet(ticker, period=period, limit=limit)
        cashflow = get_fmp_cash_flow_statement(ticker, period=period, limit=limit)
        if income is None or income.empty:
            return None, None

        info = _get_info(ticker)
        years = sorted({int(y) for y in income["year"].tolist()})
        prices = _year_end_prices(ticker, years)

        bal_by_year = {}
        if balance is not None and not balance.empty:
            for _, r in balance.iterrows():
                bal_by_year[int(r["year"])] = r

        cf_by_year = {}
        if cashflow is not None and not cashflow.empty:
            for _, r in cashflow.iterrows():
                cf_by_year[int(r["year"])] = r

        current_pb = info.get("priceToBook")
        current_ev_ebitda = info.get("enterpriseToEbitda")
        latest_year = max(years) if years else None

        ratios_records, km_records = [], []
        for _, row in income.iterrows():
            year = int(row["year"])
            eps = row.get("epsdiluted") or row.get("eps")
            net_income = row.get("netIncome")
            ebitda = row.get("ebitda")
            revenue = row.get("revenue")
            interest = row.get("interestExpense")
            bal = bal_by_year.get(year)
            cf = cf_by_year.get(year)

            price = prices.get(year)
            pe = (price / eps) if (price and eps and eps != 0) else None

            equity = bal.get("totalStockholdersEquity") if bal is not None else None
            debt = bal.get("totalDebt") if bal is not None else None
            total_assets = bal.get("totalAssets") if bal is not None else None
            cur_assets = bal.get("totalCurrentAssets") if bal is not None else None
            cur_liab = bal.get("totalCurrentLiabilities") if bal is not None else None
            op_cf = cf.get("operatingCashFlow") if cf is not None else None

            roe = (net_income / equity) if (net_income and equity and equity != 0) else None
            de = (debt / equity) if (debt is not None and equity and equity != 0) else None
            debt_ratio = (debt / total_assets) if (debt is not None and total_assets) else None
            interest_cov = (ebitda / interest) if (ebitda and interest and interest != 0) else None
            net_margin = (net_income / revenue) if (net_income and revenue and revenue != 0) else None
            current_ratio = (cur_assets / cur_liab) if (cur_assets and cur_liab and cur_liab != 0) else None
            cf_to_debt = (op_cf / debt) if (op_cf is not None and debt and debt != 0) else None

            pb = current_pb if year == latest_year else None
            ev_ebitda = current_ev_ebitda if year == latest_year else None

            ratios_records.append({
                "date": row["date"], "year": year, "calendarYear": year,
                "priceEarningsRatio": pe,
                "priceToBookRatio": pb,
                "returnOnEquity": roe,
                "debtEquityRatio": de,
                "debtRatio": debt_ratio,
                "interestCoverage": interest_cov,
                "netProfitMargin": net_margin,
                "currentRatio": current_ratio,
                "cashFlowToDebtRatio": cf_to_debt,
            })
            km_records.append({
                "date": row["date"], "year": year,
                "peRatio": pe,
                "pbRatio": pb,
                "enterpriseValueOverEBITDA": ev_ebitda,
            })

        ratios_df = pd.DataFrame(ratios_records)
        ratios_df["date"] = pd.to_datetime(ratios_df["date"])
        key_metrics_df = pd.DataFrame(km_records)
        key_metrics_df["date"] = pd.to_datetime(key_metrics_df["date"])
        return ratios_df, key_metrics_df
    except Exception as e:
        print(f"Error building ratios/key metrics for {ticker}: {e}")
        return None, None


def get_comprehensive_financial_data(ticker: str, api_key: str = None, period: str = "annual", limit: int = 5) -> dict:
    """Fetches all three financial statements plus ratios for a company."""
    print(f"Fetching comprehensive financial data for {ticker} (Yahoo Finance)...")

    financial_data = {
        'income_statement': get_fmp_income_statement(ticker, period=period, limit=limit),
        'balance_sheet': get_fmp_balance_sheet(ticker, period=period, limit=limit),
        'cash_flow': get_fmp_cash_flow_statement(ticker, period=period, limit=limit),
        'ratios': None,
        'key_metrics': None,
    }

    ratios_df, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, period=period, limit=limit)
    financial_data['ratios'] = ratios_df
    financial_data['key_metrics'] = key_metrics_df

    return financial_data


# --------------------------------------------------------------------------- #
# Peer comparison
# --------------------------------------------------------------------------- #

def combine_peer_financial_data(tickers: list[str], api_key: str = None, years_limit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combines EBITDA and EV/EBITDA for a list of peer tickers (Yahoo Finance)."""
    all_peers_data = {}
    for ticker in tickers:
        income_df = get_fmp_income_statement(ticker, limit=years_limit)
        info = _get_info(ticker)
        ev_ebitda_current = info.get("enterpriseToEbitda")

        ticker_data = {}
        if income_df is not None and not income_df.empty:
            latest_year = int(income_df["year"].max())
            for _, row in income_df.iterrows():
                year = int(row["year"])
                ticker_data.setdefault(year, {})
                ticker_data[year]["EBITDA"] = row.get("ebitda")
                if year == latest_year and ev_ebitda_current is not None:
                    ticker_data[year]["EV/EBITDA"] = ev_ebitda_current

        if ticker_data:
            all_peers_data[ticker] = ticker_data

    ebitda_records = []
    for ticker, yearly_data in all_peers_data.items():
        for year, metrics in yearly_data.items():
            if metrics.get("EBITDA") is not None:
                ebitda_records.append({"ticker": ticker, "year": year, "EBITDA": metrics["EBITDA"]})
    df_ebitda_all = pd.DataFrame(ebitda_records)
    df_ebitda_pivot = pd.DataFrame()
    if not df_ebitda_all.empty:
        df_ebitda_pivot = df_ebitda_all.pivot(index="year", columns="ticker", values="EBITDA").sort_index()

    ev_ebitda_records = []
    for ticker, yearly_data in all_peers_data.items():
        for year, metrics in yearly_data.items():
            if metrics.get("EV/EBITDA") is not None:
                ev_ebitda_records.append({"ticker": ticker, "year": year, "EV/EBITDA": metrics["EV/EBITDA"]})
    df_ev_ebitda_all = pd.DataFrame(ev_ebitda_records)
    df_ev_ebitda_pivot = pd.DataFrame()
    if not df_ev_ebitda_all.empty:
        df_ev_ebitda_pivot = df_ev_ebitda_all.pivot(index="year", columns="ticker", values="EV/EBITDA").sort_index()

    return df_ebitda_pivot, df_ev_ebitda_pivot


def project_ebitda_for_peers(df_ebitda_historical: pd.DataFrame, num_projection_years: int = 1) -> pd.DataFrame:
    """Projects EBITDA for future years based on average historical YoY growth."""
    df_projected = df_ebitda_historical.copy()
    if df_projected.empty:
        return df_projected

    last_historical_year = df_projected.index.max()

    for company in df_projected.columns:
        historical_values = df_projected[company].dropna()
        if len(historical_values) < 2:
            print(f"Not enough historical EBITDA data for {company} to project.")
            continue

        growth_rates = historical_values.pct_change().dropna()
        if growth_rates.empty or all(g == 0 for g in growth_rates):
            avg_growth_rate = 0
        else:
            avg_growth_rate = growth_rates.mean()

        current_ebitda = historical_values.iloc[-1]
        for i in range(1, num_projection_years + 1):
            projection_year = last_historical_year + i
            current_ebitda = current_ebitda * (1 + avg_growth_rate)
            df_projected.loc[projection_year, company] = current_ebitda

    return df_projected.sort_index()


# --------------------------------------------------------------------------- #
# Quote / price / profile
# --------------------------------------------------------------------------- #

def get_fmp_current_price(ticker: str, api_key: str = None) -> float | None:
    """Fetches the latest stock price from Yahoo Finance."""
    info = _get_info(ticker)
    for key in ("currentPrice", "regularMarketPrice", "regularMarketPreviousClose", "previousClose"):
        val = info.get(key)
        if val:
            return float(val)
    hist = _history(ticker, period="5d")
    if hist is not None and not hist.empty:
        return float(hist["Close"].iloc[-1])
    print(f"Price data not found for {ticker}.")
    return None


def get_analyst_insights(ticker: str, api_key: str = None) -> tuple[str | None, float | None]:
    """Fetches analyst rating and target price using Yahoo Finance."""
    rating = get_fmp_analyst_rating(ticker)
    target_price = get_fmp_target_price(ticker)
    if rating:
        print(f"[INFO] For {ticker} - Rating: {rating}")
    if target_price:
        print(f"[INFO] For {ticker} - Target Price: {target_price}")
    return rating, target_price


def get_fmp_target_price(ticker: str, api_key: str = None) -> float | None:
    """Fetches the latest analyst mean target price from Yahoo Finance."""
    info = _get_info(ticker)
    for key in ("targetMeanPrice", "targetMedianPrice"):
        val = info.get(key)
        if val:
            return float(val)
    return None


def get_fmp_analyst_rating(ticker: str, api_key: str = None) -> str | None:
    """Fetches the analyst recommendation (e.g. Buy/Hold/Sell) from Yahoo Finance."""
    info = _get_info(ticker)
    key = info.get("recommendationKey")
    if key and key != "none":
        return str(key).replace("_", " ").title()
    return None


def get_fmp_company_profile(ticker: str, api_key: str = None) -> dict | None:
    """Returns an FMP-profile-shaped dict built from Yahoo Finance .info."""
    info = _get_info(ticker)
    if not info:
        print(f"No profile data returned for {ticker}")
        return None
    exch = info.get("exchange", "")
    return {
        "symbol": ticker,
        "companyName": info.get("longName") or info.get("shortName"),
        "mktCap": info.get("marketCap"),
        "volAvg": info.get("averageVolume"),
        "beta": info.get("beta"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "exchangeShortName": _EXCHANGE_NAMES.get(exch, exch or "N/A"),
        "currency": info.get("currency", "USD"),
        "lastDiv": info.get("lastDividendValue") or info.get("trailingAnnualDividendRate"),
        "sharesOutstanding": info.get("sharesOutstanding"),
        "description": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
    }


def get_fmp_market_cap(ticker: str, api_key: str = None) -> float | None:
    """Fetches current market capitalization from Yahoo Finance."""
    info = _get_info(ticker)
    mc = info.get("marketCap")
    return float(mc) if mc else None


def get_comprehensive_company_metrics(ticker: str, api_key: str = None) -> dict:
    """Fetches all key company metrics needed for the equity report (Yahoo Finance)."""
    print(f"Fetching comprehensive company metrics for {ticker} (Yahoo Finance)...")

    info = _get_info(ticker)
    cur = info.get("currency", "USD")
    sym = _CURRENCY_SYMBOLS.get(cur, "")

    metrics = {
        'share_price': None, 'target_price': None, 'market_cap': None, 'volume': None,
        'fwd_pe': None, 'pb_ratio': None, 'dividend_yield': None, 'free_float': None,
        'roe': None, 'net_debt_to_equity': None, 'rating': None, 'beta': None,
        'sector': None, 'industry': None, 'exchange': None, '52w_range': None,
        'shares_outstanding': None, 'currency': cur, 'currency_symbol': sym,
    }

    metrics['share_price'] = get_fmp_current_price(ticker)
    metrics['target_price'] = get_fmp_target_price(ticker)
    metrics['rating'] = get_fmp_analyst_rating(ticker) or 'N/A'

    mc = info.get("marketCap")
    metrics['market_cap'] = (mc / 1e9) if mc else None  # billions
    vol = info.get("averageVolume")
    metrics['volume'] = (vol / 1e6) if vol else None  # millions
    metrics['beta'] = info.get("beta")
    metrics['sector'] = info.get("sector") or 'N/A'
    metrics['industry'] = info.get("industry") or 'N/A'
    exch = info.get("exchange", "")
    metrics['exchange'] = _EXCHANGE_NAMES.get(exch, exch or 'N/A')

    metrics['fwd_pe'] = info.get("forwardPE") or info.get("trailingPE")
    metrics['pb_ratio'] = info.get("priceToBook")

    roe = info.get("returnOnEquity")
    if roe is not None:
        metrics['roe'] = roe * 100  # to percentage

    de = info.get("debtToEquity")
    if de is not None:
        # yfinance reports debtToEquity as a percentage (e.g. 152.3); FMP used a ratio
        metrics['net_debt_to_equity'] = de / 100.0

    # Dividend yield (normalise to a percentage)
    dy = info.get("dividendYield")
    if dy is None:
        dy = info.get("trailingAnnualDividendYield")
    if dy is not None:
        metrics['dividend_yield'] = dy * 100 if dy < 1 else dy

    # 52-week range (currency aware)
    low = info.get("fiftyTwoWeekLow")
    high = info.get("fiftyTwoWeekHigh")
    if low is not None and high is not None:
        metrics['52w_range'] = f"{sym}{low:,.2f} - {sym}{high:,.2f}"

    so = info.get("sharesOutstanding")
    if so:
        metrics['shares_outstanding'] = float(so)

    ff = info.get("floatShares")
    if ff and so:
        metrics['free_float'] = round(ff / so * 100, 1)
    if metrics['free_float'] is None:
        metrics['free_float'] = 95.0

    print(f"Successfully fetched metrics for {ticker}")
    return metrics


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #

def get_technical_indicators(ticker: str, api_key: str = None) -> dict:
    """Fetch price history from Yahoo Finance and compute SMA50/200, RSI14, MACD,
    and volume signals."""
    result = {
        'sma50': None, 'sma200': None, 'rsi14': None,
        'macd': None, 'macd_signal': None, 'macd_histogram': None,
        'avg_volume_20d': None, 'latest_volume': None,
        'price': None,
        'ma_signal': 'N/A', 'rsi_signal': 'N/A',
        'macd_signal_label': 'N/A', 'volume_signal': 'N/A',
        'overall_signal': 'N/A',
    }
    try:
        hist = _history(ticker, period="1y")
        if hist is None or len(hist) < 50:
            return result

        df = hist.reset_index()
        close = df['Close'].astype(float)
        volume = df['Volume'].astype(float)

        result['price'] = close.iloc[-1]

        # SMA
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        result['sma50'] = round(sma50, 2)
        if sma200 is not None and pd.notna(sma200):
            result['sma200'] = round(sma200, 2)

        price = close.iloc[-1]
        if sma200 is not None and pd.notna(sma200):
            if price > sma50 > sma200:
                result['ma_signal'] = 'Bullish'
            elif price < sma50 < sma200:
                result['ma_signal'] = 'Bearish'
            elif price > sma200:
                result['ma_signal'] = 'Neutral-Bullish'
            else:
                result['ma_signal'] = 'Neutral-Bearish'
        elif price > sma50:
            result['ma_signal'] = 'Bullish'
        else:
            result['ma_signal'] = 'Bearish'

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        result['rsi14'] = round(rsi_val, 1)
        if rsi_val > 70:
            result['rsi_signal'] = 'Overbought'
        elif rsi_val < 30:
            result['rsi_signal'] = 'Oversold'
        elif rsi_val > 55:
            result['rsi_signal'] = 'Bullish'
        elif rsi_val < 45:
            result['rsi_signal'] = 'Bearish'
        else:
            result['rsi_signal'] = 'Neutral'

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        result['macd'] = round(macd_line.iloc[-1], 2)
        result['macd_signal'] = round(signal_line.iloc[-1], 2)
        result['macd_histogram'] = round(histogram.iloc[-1], 2)
        if macd_line.iloc[-1] > signal_line.iloc[-1] and histogram.iloc[-1] > 0:
            result['macd_signal_label'] = 'Bullish'
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and histogram.iloc[-1] < 0:
            result['macd_signal_label'] = 'Bearish'
        else:
            result['macd_signal_label'] = 'Neutral'

        # Volume
        avg_vol = volume.tail(20).mean()
        latest_vol = volume.iloc[-1]
        result['avg_volume_20d'] = round(avg_vol)
        result['latest_volume'] = round(latest_vol)
        vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1
        if vol_ratio > 1.5:
            result['volume_signal'] = 'High Activity'
        elif vol_ratio < 0.5:
            result['volume_signal'] = 'Low Activity'
        else:
            result['volume_signal'] = 'Normal'

        # Overall signal
        signals = [result['ma_signal'], result['rsi_signal'], result['macd_signal_label']]
        bullish = sum(1 for s in signals if 'Bullish' in s or s == 'Oversold')
        bearish = sum(1 for s in signals if 'Bearish' in s or s == 'Overbought')
        if bullish >= 2:
            result['overall_signal'] = 'Bullish'
        elif bearish >= 2:
            result['overall_signal'] = 'Bearish'
        else:
            result['overall_signal'] = 'Neutral'

        print(f"✅ Computed technical indicators for {ticker}: {result['overall_signal']}")
    except Exception as e:
        print(f"⚠️ Could not compute technical indicators: {e}")
    return result


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #

def get_company_news(ticker: str, api_key: str = None, days_back: int = 5, limit: int = 50) -> list[dict] | None:
    """Fetches recent company news from Yahoo Finance.

    Returns a list of dicts with FMP-compatible fields:
    symbol, title, publishedDate, text, site, url.
    """
    try:
        raw = _get_ticker(ticker).news or []
        if not raw:
            print(f"No news data returned from Yahoo Finance for {ticker}.")
            return None

        filtered_news = []
        for article in raw[:limit]:
            # yfinance changed its news schema; support both old and new shapes.
            content = article.get("content") if isinstance(article.get("content"), dict) else None
            if content:
                title = content.get("title")
                summary = content.get("summary") or content.get("description")
                pub_date = content.get("pubDate") or content.get("displayTime")
                provider = (content.get("provider") or {}).get("displayName")
                url = ((content.get("canonicalUrl") or {}) or {}).get("url") or \
                      ((content.get("clickThroughUrl") or {}) or {}).get("url")
            else:
                title = article.get("title")
                summary = article.get("summary")
                ts = article.get("providerPublishTime")
                pub_date = (
                    datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    if ts else None
                )
                provider = article.get("publisher")
                url = article.get("link")

            filtered_news.append({
                "symbol": ticker,
                "title": title,
                "publishedDate": pub_date,
                "text": summary or title,
                "site": provider,
                "url": url,
            })

        print(f"Successfully fetched {len(filtered_news)} news articles for {ticker}")
        return filtered_news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return None


if __name__ == "__main__":
    print("Testing market_data_api.py (Yahoo Finance edition)...")
    test_ticker = os.environ.get("TEST_TICKER", "ADANIENT.NS")
    print(f"\nTesting get_comprehensive_financial_data for {test_ticker}...")
    financial_data = get_comprehensive_financial_data(test_ticker)
    for statement_type, df in financial_data.items():
        if df is not None and not df.empty:
            print(f"{statement_type}: {len(df)} rows of data")
        else:
            print(f"{statement_type}: No data")

    print(f"\nTesting get_comprehensive_company_metrics for {test_ticker}...")
    m = get_comprehensive_company_metrics(test_ticker)
    print({k: m[k] for k in ('share_price', 'market_cap', 'fwd_pe', 'pb_ratio', 'roe', '52w_range', 'currency')})

    print("\nmarket_data_api.py tests complete.")
