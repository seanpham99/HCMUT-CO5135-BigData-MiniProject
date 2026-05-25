"""
Mock utilities for vnstock library.

This module provides mock classes for vnstock Quote, Finance, and Company
to avoid making real API calls during tests.
"""

import pandas as pd
from unittest.mock import Mock


class MockQuote:
    """Mock vnstock Quote class for stock price data."""
    
    def __init__(self, symbol):
        self.symbol = symbol
    
    def history(self, start, end, interval='1D'):
        """
        Mock history method that returns sample OHLCV data.
        
        Args:
            start: Start date
            end: End date
            interval: Time interval (default '1D')
            
        Returns:
            pd.DataFrame: Mock stock price data
        """
        # Return empty DataFrame for invalid symbols
        if self.symbol == 'INVALID':
            return pd.DataFrame()
        
        # Generate sample data
        dates = pd.date_range(start=start, end=end, freq='D')
        
        df = pd.DataFrame({
            'time': dates,
            'open': [100.0 + i * 0.5 for i in range(len(dates))],
            'high': [101.0 + i * 0.5 for i in range(len(dates))],
            'low': [99.0 + i * 0.5 for i in range(len(dates))],
            'close': [100.5 + i * 0.5 for i in range(len(dates))],
            'volume': [1000000 + i * 10000 for i in range(len(dates))],
        })
        
        return df


class MockFinance:
    """Mock vnstock Finance class for financial data."""
    
    def __init__(self, symbol):
        self.symbol = symbol
    
    def ratio(self, period='quarter', dropna=True):
        """
        Mock ratio method that returns financial ratios.
        
        Args:
            period: 'quarter' or 'year'
            dropna: Whether to drop NaN values
            
        Returns:
            pd.DataFrame: Mock financial ratios with MultiIndex columns
        """
        if self.symbol == 'INVALID':
            return pd.DataFrame()
        
        # Mock data matching real vnstock API structure
        data = {
            ('Chỉ số định giá', 'P/E'): [15.5, 16.2, 14.8],
            ('Chỉ số định giá', 'P/B'): [2.1, 2.3, 2.0],
            ('Chỉ số định giá', 'P/S'): [1.5, 1.6, 1.4],
            ('Chỉ số định giá', 'Market Capital'): [50000, 51000, 49000],
            ('Chỉ số hiệu quả hoạt động', 'ROE (%)'): [18.5, 19.2, 17.8],
            ('Chỉ số hiệu quả hoạt động', 'ROA (%)'): [8.5, 9.0, 8.2],
            ('Chỉ số hiệu quả hoạt động', 'ROIC (%)'): [12.5, 13.1, 12.0],
            ('Chỉ số tăng trưởng', 'EPS'): [5000, 5200, 4900],
            ('Chỉ số tăng trưởng', 'BVPS'): [25000, 26000, 24500],
            ('Chỉ số nợ', 'Debt/Equity'): [0.5, 0.48, 0.52],
            ('Năm báo cáo', 'yearReport'): [2024, 2024, 2023],
            ('Độ dài báo cáo', 'lengthReport'): [4, 3, 4],
        }
        
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        return df
    
    def income_statement(self, period='quarter', dropna=True):
        """
        Mock income_statement method that returns P&L data.
        
        Args:
            period: 'quarter' or 'year'
            dropna: Whether to drop NaN values
            
        Returns:
            pd.DataFrame: Mock income statement with MultiIndex columns
        """
        if self.symbol == 'INVALID':
            return pd.DataFrame()
        
        data = {
            ('Doanh thu thuần', ''): [500000000000, 480000000000, 520000000000],
            ('Giá vốn hàng bán', ''): [350000000000, 340000000000, 360000000000],
            ('Lợi nhuận gộp', ''): [150000000000, 140000000000, 160000000000],
            ('Lợi nhuận hoạt động', ''): [80000000000, 75000000000, 85000000000],
            ('Lợi nhuận sau thuế', ''): [60000000000, 55000000000, 65000000000],
            ('Năm', ''): [2024, 2024, 2024],
            ('Quý', ''): [4, 3, 2],
        }
        
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        return df


class MockCompany:
    """Mock vnstock Company class for company data."""
    
    def __init__(self, symbol):
        self.symbol = symbol
    
    def dividends(self):
        """
        Mock dividends method that returns dividend history.
        
        Returns:
            pd.DataFrame: Mock dividend data
        """
        if self.symbol == 'INVALID':
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'exerciseDate': ['2024-06-15', '2023-06-15', '2022-06-15'],
            'issueDate': ['2024-05-01', '2023-05-01', '2022-05-01'],
            'rightName': ['Cổ tức tiền mặt', 'Cổ tức tiền mặt', 'Cổ tức cổ phiếu'],
            'value': [1500, 1400, 10],
        })
        
        return df
    
    def news(self, limit=50):
        """
        Mock news method that returns news articles.
        
        Args:
            limit: Maximum number of articles
            
        Returns:
            pd.DataFrame: Mock news data
        """
        if self.symbol == 'INVALID':
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'title': [
                f'{self.symbol} công bố kết quả kinh doanh quý 4',
                f'Tin tức về {self.symbol}',
                f'{self.symbol} tăng trưởng mạnh',
            ][:limit],
            'source': ['CafeF', 'VnExpress', 'Đầu tư'][:limit],
            'publishDate': ['2024-12-20', '2024-12-19', '2024-12-18'][:limit],
        })
        
        return df
