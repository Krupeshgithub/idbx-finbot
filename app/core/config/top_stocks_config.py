"""
Configuration for Top Stocks / Rankings Feature
================================================
Centralized configuration for all top stocks related settings.
"""
from typing import List


class TopStocksConfig:
    """Configuration for top stocks feature."""
    
    # Default limits
    DEFAULT_LIMIT: int = 10
    MAX_LIMIT: int = 25
    MIN_LIMIT: int = 1
    
    # Fetch multiplier (fetch 2x to handle errors)
    FETCH_MULTIPLIER: int = 2
    
    # Caching (future enhancement)
    CACHE_ENABLED: bool = False
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # US Large-Cap Stocks (Top ~50 by typical market cap)
    # This list contains TICKER SYMBOLS ONLY (not market data)
    # All market data (prices, market caps, volumes) is fetched LIVE from Alpha Vantage API
    # 
    # IMPORTANT: This is NOT hardcoded market data - it's just a list of company identifiers
    # Think of it like a phone book: names are static, but phone numbers (data) are fetched live
    # 
    # Why we need this list:
    # - Alpha Vantage doesn't have a "get top stocks" endpoint
    # - We need to know WHICH companies to query for their live data
    # - This list is updated periodically (monthly) to reflect market changes
    # 
    # How it works:
    # 1. Use this list to know which tickers to query
    # 2. Fetch LIVE market cap, price, volume from API for each ticker
    # 3. Sort by LIVE market cap (not hardcoded values)
    # 4. Return top N based on CURRENT data
    US_MEGA_CAPS: List[str] = [
        # Tech Giants (Trillion+ market cap range)
        "AAPL",   # Apple Inc.
        "MSFT",   # Microsoft Corporation
        "GOOGL",  # Alphabet Inc. Class A
        "GOOG",   # Alphabet Inc. Class C
        "AMZN",   # Amazon.com Inc.
        "NVDA",   # NVIDIA Corporation
        "META",   # Meta Platforms (Facebook)
        "TSLA",   # Tesla Inc.
        "AVGO",   # Broadcom Inc.
        "ORCL",   # Oracle Corporation
        "ADBE",   # Adobe Inc.
        "CRM",    # Salesforce Inc.
        "NFLX",   # Netflix Inc.
        "INTC",   # Intel Corporation
        "AMD",    # Advanced Micro Devices
        "CSCO",   # Cisco Systems
        "QCOM",   # Qualcomm Inc.
        
        # Financial Services
        "BRK.B",  # Berkshire Hathaway Class B
        "V",      # Visa Inc.
        "JPM",    # JPMorgan Chase & Co.
        "MA",     # Mastercard Inc.
        "BAC",    # Bank of America Corp.
        "WFC",    # Wells Fargo & Company
        "MS",     # Morgan Stanley
        "GS",     # Goldman Sachs Group
        "AXP",    # American Express Company
        "BLK",    # BlackRock Inc.
        "C",      # Citigroup Inc.
        
        # Healthcare & Pharma
        "UNH",    # UnitedHealth Group
        "JNJ",    # Johnson & Johnson
        "LLY",    # Eli Lilly and Company
        "ABBV",   # AbbVie Inc.
        "MRK",    # Merck & Co.
        "PFE",    # Pfizer Inc.
        "TMO",    # Thermo Fisher Scientific
        "ABT",    # Abbott Laboratories
        "DHR",    # Danaher Corporation
        "BMY",    # Bristol-Myers Squibb
        
        # Consumer & Retail
        "WMT",    # Walmart Inc.
        "PG",     # Procter & Gamble
        "HD",     # Home Depot
        "COST",   # Costco Wholesale
        "KO",     # Coca-Cola Company
        "PEP",    # PepsiCo Inc.
        "MCD",    # McDonald's Corporation
        "NKE",    # Nike Inc.
        "SBUX",   # Starbucks Corporation
        "TGT",    # Target Corporation
        
        # Energy
        "XOM",    # Exxon Mobil Corporation
        "CVX",    # Chevron Corporation
        "COP",    # ConocoPhillips
        "SLB",    # Schlumberger Limited
        
        # Industrial & Manufacturing
        "BA",     # Boeing Company
        "CAT",    # Caterpillar Inc.
        "GE",     # General Electric
        "HON",    # Honeywell International
        "UPS",    # United Parcel Service
        
        # Telecom & Media
        "T",      # AT&T Inc.
        "VZ",     # Verizon Communications
        "CMCSA",  # Comcast Corporation
        "DIS",    # Walt Disney Company
    ]
    
    # Sector mappings for future sector-specific queries
    SECTOR_TICKERS = {
        "technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO"],
        "financial": ["BRK.B", "V", "JPM", "MA"],
        "healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
        "consumer": ["WMT", "PG", "HD", "COST", "KO", "PEP"],
        "energy": ["XOM", "CVX"],
    }
    
    # Response formatting
    INCLUDE_SECTOR_IN_TABLE: bool = True
    INCLUDE_PE_RATIO: bool = False  # Can be slow, disabled by default
    INCLUDE_DIVIDEND_YIELD: bool = False
    
    # Error handling
    MIN_SUCCESSFUL_FETCHES: int = 5  # Minimum stocks to return valid response
    RETRY_ON_ERROR: bool = False
    MAX_RETRIES: int = 2
    
    # Performance
    PARALLEL_FETCHING: bool = False  # Future enhancement
    MAX_CONCURRENT_REQUESTS: int = 5
    
    @classmethod
    def get_tickers_for_sector(cls, sector: str) -> List[str]:
        """Get tickers for a specific sector."""
        sector_lower = sector.lower()
        return cls.SECTOR_TICKERS.get(sector_lower, cls.US_MEGA_CAPS)
    
    @classmethod
    def validate_limit(cls, limit: int) -> int:
        """Validate and clamp limit to acceptable range."""
        return max(cls.MIN_LIMIT, min(limit, cls.MAX_LIMIT))
    
    @classmethod
    def get_fetch_count(cls, limit: int) -> int:
        """Calculate how many tickers to fetch (with buffer for errors)."""
        validated_limit = cls.validate_limit(limit)
        fetch_count = validated_limit * cls.FETCH_MULTIPLIER
        return min(fetch_count, len(cls.US_MEGA_CAPS))


# Singleton instance
top_stocks_config = TopStocksConfig()
