"""
Pytest configuration and shared fixtures for Portfolios Tracker tests.

This module provides common fixtures used across all tests:
- Mock environment variables
- Sample DataFrames
- Mock API responses
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock


@pytest.fixture(scope="session", autouse=True)
def mock_environment_variables():
    """
    Set up mock environment variables for all tests.
    Runs once per test session and cleans up after.
    """
    # Store original values
    original_env = {}
    env_vars = {
        "SUPABASE_DB_URL": "postgresql://test_user:test_password@localhost:5432/postgres",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SECRET_OR_SERVICE_ROLE_KEY": "test-service-role-key",
        "TELEGRAM_BOT_TOKEN": "test_token_123456",
        "TELEGRAM_CHAT_ID": "test_chat_id_789",
        "GEMINI_API_KEY": "test_gemini_key_abc",
    }

    # Save originals and set test values
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield env_vars

    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def sample_stock_price_df():
    """
    Provides a sample stock price DataFrame with technical indicators.

    Returns:
        pd.DataFrame: Sample OHLCV data with indicators
    """
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")

    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]
    for _ in range(249):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame(
        {
            "time": dates,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, 250),
        }
    )

    return df


@pytest.fixture
def sample_financial_ratios_df():
    """
    Provides a sample financial ratios DataFrame.

    Returns:
        pd.DataFrame: Sample quarterly financial ratios
    """
    data = {
        ("Chỉ số định giá", "P/E"): [15.5, 16.2, 14.8, 15.9],
        ("Chỉ số định giá", "P/B"): [2.1, 2.3, 2.0, 2.2],
        ("Chỉ số định giá", "P/S"): [1.5, 1.6, 1.4, 1.5],
        ("Chỉ số hiệu quả hoạt động", "ROE (%)"): [18.5, 19.2, 17.8, 18.9],
        ("Chỉ số hiệu quả hoạt động", "ROA (%)"): [8.5, 9.0, 8.2, 8.7],
        ("Chỉ số hiệu quả hoạt động", "ROIC (%)"): [12.5, 13.1, 12.0, 12.8],
        ("Chỉ số tăng trưởng", "EPS"): [5000, 5200, 4900, 5100],
        ("Chỉ số tăng trưởng", "BVPS"): [25000, 26000, 24500, 25500],
        ("Năm", ""): [2024, 2024, 2023, 2023],
        ("Quý", ""): [4, 3, 4, 3],
    }

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df


@pytest.fixture
def sample_income_statement_df():
    """
    Provides a sample income statement DataFrame.

    Returns:
        pd.DataFrame: Sample quarterly income statement
    """
    data = {
        ("Doanh thu thuần", ""): [500000000000, 480000000000, 520000000000],
        ("Giá vốn hàng bán", ""): [350000000000, 340000000000, 360000000000],
        ("Lợi nhuận gộp", ""): [150000000000, 140000000000, 160000000000],
        ("Lợi nhuận hoạt động", ""): [80000000000, 75000000000, 85000000000],
        ("Lợi nhuận sau thuế", ""): [60000000000, 55000000000, 65000000000],
        ("Năm", ""): [2024, 2024, 2024],
        ("Quý", ""): [4, 3, 2],
    }

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df


@pytest.fixture
def sample_dividends_df():
    """
    Provides a sample dividends DataFrame.

    Returns:
        pd.DataFrame: Sample dividend history
    """
    df = pd.DataFrame(
        {
            "exerciseDate": ["2024-06-15", "2023-06-15", "2022-06-15"],
            "issueDate": ["2024-05-01", "2023-05-01", "2022-05-01"],
            "rightName": ["Cổ tức tiền mặt", "Cổ tức tiền mặt", "Cổ tức cổ phiếu"],
            "value": [1500, 1400, 10],  # VND or percentage
        }
    )

    return df


@pytest.fixture
def sample_news_df():
    """
    Provides a sample news DataFrame.

    Returns:
        pd.DataFrame: Sample news articles
    """
    df = pd.DataFrame(
        {
            "title": [
                "HPG công bố kết quả kinh doanh quý 4 tăng trưởng mạnh",
                "Thị trường chứng khoán biến động mạnh do yếu tố vĩ mô",
                "VCB tăng vốn điều lệ, mở rộng hoạt động",
            ],
            "source": ["CafeF", "VnExpress", "Đầu tư"],
            "publish_date": [
                datetime(2024, 12, 20, 10, 30),
                datetime(2024, 12, 19, 14, 15),
                datetime(2024, 12, 18, 9, 45),
            ],
            "price_impact_1d": [2.5, -1.2, 1.8],
            "price_impact_7d": [5.3, -0.5, 3.2],
        }
    )

    return df


@pytest.fixture
def sample_airflow_context():
    """
    Provides a sample Airflow task context for callback testing.

    Returns:
        dict: Mock Airflow context
    """
    context = {
        "dag": Mock(dag_id="test_dag"),
        "task": Mock(task_id="test_task"),
        "execution_date": datetime(2024, 12, 22, 10, 0, 0),
        "run_id": "test_run_123",
        "task_instance": Mock(
            dag_id="test_dag",
            task_id="test_task",
            execution_date=datetime(2024, 12, 22, 10, 0, 0),
            state="success",
            duration=120.5,
        ),
    }

    return context


@pytest.fixture
def mock_telegram_response():
    """
    Provides a mock successful Telegram API response.

    Returns:
        dict: Mock response data
    """
    return {
        "ok": True,
        "result": {
            "message_id": 12345,
            "chat": {"id": 789, "type": "private"},
            "date": 1703260800,
            "text": "Test message",
        },
    }


@pytest.fixture
def mock_gemini_response():
    """
    Provides a mock successful Gemini AI API response.

    Returns:
        dict: Mock response data
    """
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "**Tổng quan thị trường:**\n\nThị trường có xu hướng tích cực..."
                        }
                    ]
                }
            }
        ]
    }
