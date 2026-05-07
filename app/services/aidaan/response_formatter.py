"""
Response formatting utilities for AIDANN.

Provides institutional-grade response formatting with mandatory data provenance,
structured insights, and professional presentation.
"""

from typing import List, Optional, Dict, Any


def _normalize_data_sources(data_source: str | List[str]) -> str:
    if isinstance(data_source, list):
        cleaned = [source.strip() for source in data_source if source and source.strip()]
        return ", ".join(dict.fromkeys(cleaned))
    return str(data_source).strip()


def format_response_with_provenance(
    content: str,
    data_source: str | List[str],
    market_insight: Optional[str] = None,
    trade_implication: Optional[str] = None,
    key_insights: Optional[List[str]] = None,
    similarity_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format AIDANN response with mandatory data provenance.
    
    This ensures transparency and trust for institutional traders by clearly
    stating the source of all information.
    
    Args:
        content: Main response content
        data_source: Source of the data (e.g., "Alpha Vantage Live Feed", 
                    "IDBX Corporate Knowledge Base v1.0", "IDBX Historical Database")
        market_insight: Optional market analysis section
        trade_implication: Optional trading implications section
        key_insights: Optional list of bullet-point insights
        similarity_score: Optional similarity score for semantic search results
        metadata: Optional additional metadata to include
    
    Returns:
        Formatted response string with data provenance
        
    Example:
        response = format_response_with_provenance(
            content="AAPL is trading at $175.23...",
            data_source="Alpha Vantage Live Feed",
            market_insight="Apple stock showing bullish momentum...",
            trade_implication="Consider long positions with tight stops...",
            key_insights=[
                "Volume above 20-day average",
                "RSI at 62 (neutral to bullish)",
                "Breaking resistance at $174"
            ]
        )
    """
    response_parts = []
    normalized_source = _normalize_data_sources(data_source)
    
    # Market Insight
    if market_insight:
        response_parts.append(f"**Market Insight:**\n{market_insight}\n\n")
    
    # Trade Implication
    if trade_implication:
        response_parts.append(f"**Trade Implication:**\n{trade_implication}\n\n")
    
    # Key Insights
    if key_insights:
        response_parts.append("**Key Insights:**\n")
        for insight in key_insights:
            response_parts.append(f"• {insight}\n")
        response_parts.append("\n")
    
    # Main Content
    response_parts.append(content)
    
    # Additional Metadata (optional, non-technical presentation only)
    if metadata and metadata.get("display_additional_info"):
        response_parts.append("\n\n**Additional Information:**\n")
        for key, value in metadata.items():
            if key == "display_additional_info":
                continue
            response_parts.append(f"• {key}: {value}\n")

    # Mandatory provenance footer for transparency.
    response_parts.append(f"\n\n**IDBX Data Provenance:** {normalized_source}")
    
    return "".join(response_parts)


def format_corporate_knowledge_response(
    results: List[Dict[str, Any]],
    query: str
) -> str:
    """
    Format corporate knowledge search results into a professional response.
    
    Args:
        results: List of search results from search_corporate_knowledge()
        query: Original user query
    
    Returns:
        Formatted response with data provenance
    """
    if not results:
        return format_response_with_provenance(
            content=f"I don't have specific information about '{query}' in my corporate knowledge base. "
                   "Please contact your IDBX administrator for more details.",
            data_source="IDBX Corporate Knowledge Base v1.0"
        )
    
    # Use the top result
    top_result = results[0]
    
    # Format the main content
    content = top_result['content']
    
    # Extract metadata
    metadata_json = top_result.get('metadata_json', {})
    
    # Keep corporate answers concise and non-technical for end users.
    _ = metadata_json

    return format_response_with_provenance(
        content=content,
        data_source="IDBX Corporate Knowledge Base v1.0",
        key_insights=None,
        similarity_score=None
    )


def format_market_data_response(
    ticker: str,
    price: float,
    change: float,
    change_percent: float,
    volume: Optional[int] = None,
    market_insight: Optional[str] = None,
    trade_implication: Optional[str] = None,
    key_insights: Optional[List[str]] = None,
    data_source: str = "Alpha Vantage Live Feed"
) -> str:
    """
    Format market data response with institutional-grade structure.
    
    Args:
        ticker: Stock ticker symbol
        price: Current price
        change: Price change
        change_percent: Percentage change
        volume: Trading volume
        market_insight: Market analysis
        trade_implication: Trading implications
        key_insights: List of key insights
        data_source: Data source name
    
    Returns:
        Formatted response with data provenance
    """
    # Build main content
    change_direction = "up" if change >= 0 else "down"
    change_symbol = "+" if change >= 0 else ""
    
    content = f"""**{ticker}** is currently trading at **${price:.2f}**, {change_direction} {change_symbol}${change:.2f} ({change_symbol}{change_percent:.2f}%) today."""
    
    if volume:
        content += f"\n\nTrading Volume: {volume:,} shares"
    
    return format_response_with_provenance(
        content=content,
        data_source=data_source,
        market_insight=market_insight,
        trade_implication=trade_implication,
        key_insights=key_insights
    )


def format_error_response(
    error_message: str,
    data_source: str = "AIDANN System",
    suggestion: Optional[str] = None
) -> str:
    """
    Format error responses with data provenance.
    
    Args:
        error_message: Error message to display
        data_source: Source of the error
        suggestion: Optional suggestion for resolution
    
    Returns:
        Formatted error response
    """
    content = f"⚠️ {error_message}"
    
    if suggestion:
        content += f"\n\n**Suggestion:** {suggestion}"
    
    return format_response_with_provenance(
        content=content,
        data_source=data_source
    )


# Data source constants for consistency
class DataSource:
    """Standard data source identifiers for AIDANN responses."""
    
    ALPHA_VANTAGE_LIVE = "Alpha Vantage Live Feed"
    IDBX_HISTORICAL = "IDBX Historical Database"
    IDBX_CORPORATE_KB = "IDBX Corporate Knowledge Base v1.0"
    LSEG_LIVE = "LSEG Real-Time Market Data"
    FINBERT_SENTIMENT = "FinBERT Sentiment Analysis Engine"
    AIDANN_SYSTEM = "AIDANN System"
    BIGQUERY_ANALYTICS = "IDBX BigQuery Analytics"
