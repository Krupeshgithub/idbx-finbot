"""
Calculation Agent - Financial Mathematics & Risk Metrics
========================================================
Handles portfolio calculations, risk metrics, and optimization.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from app.services.aidaan.core.base import BaseAgent
from app.schemas.aidaan import AidaanMessageResponse
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class CalculationAgent(BaseAgent):
    """
    Specialized agent for financial calculations and risk metrics.
    """

    def __init__(self):
        super().__init__(
            name="calculation",
            model_name=settings.VERTEX_AI_REASONING_MODEL_NAME  # Use Pro for complex math
        )

    def get_capabilities(self) -> List[str]:
        """
        Returns a list of high-level features provided by this agent.
        """
        return [
            "Calculate portfolio returns and volatility",
            "Compute risk metrics (Sharpe ratio, Beta, VaR)",
            "Analyze maximum drawdown",
            "Optimize asset allocation",
            "Provide rebalancing recommendations"
        ]

    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate daily returns from price series."""
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        return returns

    @staticmethod
    def calculate_volatility(returns: List[float], annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns, ddof=1)
        
        if annualize:
            # Annualize assuming 252 trading days
            volatility = volatility * np.sqrt(252)
        
        return float(volatility)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.05,  # 5% default
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio.
        Sharpe = (Mean Return - Risk Free Rate) / Volatility
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
        
        if volatility == 0:
            return 0.0
        
        if annualize:
            # Annualize returns and volatility
            mean_return = mean_return * 252
            volatility = volatility * np.sqrt(252)
        
        sharpe = (mean_return - risk_free_rate) / volatility
        return float(sharpe)

    @staticmethod
    def calculate_beta(
        asset_returns: List[float],
        market_returns: List[float]
    ) -> float:
        """
        Calculate Beta (systematic risk).
        Beta = Covariance(asset, market) / Variance(market)
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0  # Default to market beta
        
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns, ddof=1)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return float(beta)

    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Dict[str, float]:
        """
        Calculate maximum drawdown.
        Max Drawdown = (Trough - Peak) / Peak
        """
        if len(prices) < 2:
            return {"max_drawdown": 0.0, "peak": 0.0, "trough": 0.0}
        
        peak = prices[0]
        max_dd = 0.0
        peak_price = peak
        trough_price = peak
        
        for price in prices:
            if price > peak:
                peak = price
            
            drawdown = (price - peak) / peak
            if drawdown < max_dd:
                max_dd = drawdown
                peak_price = peak
                trough_price = price
        
        return {
            "max_drawdown": float(abs(max_dd)),
            "max_drawdown_pct": float(abs(max_dd) * 100),
            "peak": float(peak_price),
            "trough": float(trough_price)
        }

    @staticmethod
    def calculate_var(
        returns: List[float],
        confidence_level: float = 0.95,
        portfolio_value: float = 1000000
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR).
        VaR = Portfolio Value * Percentile of Returns
        """
        if len(returns) < 10:
            return {"var": 0.0, "var_pct": 0.0}
        
        # Sort returns
        sorted_returns = sorted(returns)
        
        # Find percentile
        index = int((1 - confidence_level) * len(sorted_returns))
        var_return = sorted_returns[index]
        
        var_amount = portfolio_value * abs(var_return)
        
        return {
            "var": float(var_amount),
            "var_pct": float(abs(var_return) * 100),
            "confidence_level": confidence_level
        }

    @staticmethod
    def calculate_portfolio_metrics(
        holdings: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics.
        
        holdings format:
        {
            "HDFC": {
                "weight": 0.35,
                "prices": [100, 102, 101, ...],
                "beta": 1.2
            },
            ...
        }
        """
        total_return = 0.0
        total_volatility = 0.0
        weighted_beta = 0.0
        portfolio_returns = []
        
        for ticker, data in holdings.items():
            weight = data.get("weight", 0.0)
            prices = data.get("prices", [])
            beta = data.get("beta", 1.0)
            
            if len(prices) < 2:
                continue
            
            # Calculate returns
            returns = CalculationAgent.calculate_returns(prices)
            
            if not returns:
                continue
            
            # Asset metrics
            asset_return = np.mean(returns) * 252  # Annualized
            asset_volatility = CalculationAgent.calculate_volatility(returns)
            
            # Weighted contributions
            total_return += weight * asset_return
            total_volatility += (weight ** 2) * (asset_volatility ** 2)
            weighted_beta += weight * beta
            
            # Portfolio daily returns
            if not portfolio_returns:
                portfolio_returns = [r * weight for r in returns]
            else:
                for i, r in enumerate(returns):
                    if i < len(portfolio_returns):
                        portfolio_returns[i] += r * weight
        
        # Portfolio volatility (simplified - assumes no correlation)
        portfolio_volatility = np.sqrt(total_volatility)
        
        # Sharpe ratio
        sharpe = CalculationAgent.calculate_sharpe_ratio(portfolio_returns) if portfolio_returns else 0.0
        
        # Max drawdown
        if portfolio_returns:
            # Convert returns to price series
            portfolio_prices = [portfolio_value]
            for ret in portfolio_returns:
                portfolio_prices.append(portfolio_prices[-1] * (1 + ret))
            
            drawdown_metrics = CalculationAgent.calculate_max_drawdown(portfolio_prices)
        else:
            drawdown_metrics = {"max_drawdown_pct": 0.0}
        
        # VaR
        var_metrics = CalculationAgent.calculate_var(
            portfolio_returns,
            confidence_level=0.95,
            portfolio_value=portfolio_value
        ) if portfolio_returns else {"var": 0.0, "var_pct": 0.0}
        
        return {
            "expected_return": float(total_return * 100),  # Percentage
            "volatility": float(portfolio_volatility * 100),  # Percentage
            "sharpe_ratio": float(sharpe),
            "beta": float(weighted_beta),
            "max_drawdown_pct": float(drawdown_metrics.get("max_drawdown_pct", 0.0)),
            "var_95": float(var_metrics.get("var", 0.0)),
            "var_95_pct": float(var_metrics.get("var_pct", 0.0))
        }

    @staticmethod
    def optimize_allocation(
        current_allocation: Dict[str, float],
        target_return: float,
        max_drawdown: float,
        asset_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Suggest optimal allocation based on constraints.
        
        This is a simplified optimization. For production, use scipy.optimize or cvxpy.
        """
        recommendations = []
        
        # Calculate current portfolio metrics
        current_return = sum(
            weight * asset_metrics.get(ticker, {}).get("return", 0.0)
            for ticker, weight in current_allocation.items()
        )
        
        current_risk = sum(
            weight * asset_metrics.get(ticker, {}).get("volatility", 0.0)
            for ticker, weight in current_allocation.items()
        )
        
        # Check if target is achievable
        if current_return < target_return:
            recommendations.append({
                "action": "increase_equity",
                "reason": f"Current return {current_return:.2f}% < Target {target_return:.2f}%",
                "suggestion": "Increase allocation to higher-return assets"
            })
        
        if current_risk > max_drawdown:
            recommendations.append({
                "action": "reduce_risk",
                "reason": f"Current risk {current_risk:.2f}% > Max drawdown {max_drawdown:.2f}%",
                "suggestion": "Increase allocation to defensive assets (gold ETF, liquid fund)"
            })
        
        # Sector concentration check
        equity_allocation = sum(
            weight for ticker, weight in current_allocation.items()
            if ticker not in ["gold_etf", "liquid_fund"]
        )
        
        if equity_allocation > 0.80:
            recommendations.append({
                "action": "diversify",
                "reason": f"Equity concentration {equity_allocation*100:.1f}% is high",
                "suggestion": "Consider increasing defensive allocation"
            })
        
        return {
            "current_return": float(current_return),
            "target_return": float(target_return),
            "current_risk": float(current_risk),
            "max_acceptable_risk": float(max_drawdown),
            "rebalancing_needed": len(recommendations) > 0,
            "recommendations": recommendations
        }

    async def handle_message(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        """
        Handle calculation requests.
        """
        logger.info(f"[CalculationAgent] Processing calculation request: {text[:100]}...")
        
        context = context or {}
        
        # Extract calculation parameters from context
        # (These would be provided by the coordinator/market agent)
        holdings = context.get("holdings", {})
        portfolio_value = context.get("portfolio_value", 1850000)
        target_return = context.get("target_return", 0.12)
        max_drawdown = context.get("max_drawdown", 0.08)
        
        if not holdings:
            return self.build_message_response(
                reply="मुझे portfolio holdings की जानकारी नहीं मिली। कृपया holdings data provide करें।",
                bullets=[
                    "Holdings data required",
                    "Format: {ticker: {weight, prices, beta}}"
                ],
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
                latency_ms=0
            )
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(holdings, portfolio_value)
        
        # Optimization
        current_allocation = {
            ticker: data.get("weight", 0.0)
            for ticker, data in holdings.items()
        }
        
        asset_metrics = {
            ticker: {
                "return": np.mean(self.calculate_returns(data.get("prices", []))) * 252 * 100,
                "volatility": self.calculate_volatility(self.calculate_returns(data.get("prices", []))) * 100
            }
            for ticker, data in holdings.items()
        }
        
        optimization = self.optimize_allocation(
            current_allocation,
            target_return * 100,
            max_drawdown * 100,
            asset_metrics
        )
        
        # Build response
        reply = f"""
**Portfolio Risk & Return Analysis**

**Current Metrics:**
- Expected Annual Return: {portfolio_metrics['expected_return']:.2f}%
- Portfolio Volatility: {portfolio_metrics['volatility']:.2f}%
- Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}
- Portfolio Beta: {portfolio_metrics['beta']:.2f}
- Maximum Drawdown: {portfolio_metrics['max_drawdown_pct']:.2f}%
- Value at Risk (95%): ₹{portfolio_metrics['var_95']:,.0f} ({portfolio_metrics['var_95_pct']:.2f}%)

**Target vs Actual:**
- Target Return: {target_return*100:.0f}% | Current: {portfolio_metrics['expected_return']:.2f}%
- Max Acceptable Drawdown: {max_drawdown*100:.0f}% | Current: {portfolio_metrics['max_drawdown_pct']:.2f}%

**Rebalancing Recommendation:**
{"✅ Portfolio is well-balanced" if not optimization['rebalancing_needed'] else "⚠️ Rebalancing suggested"}
"""
        
        bullets = []
        for rec in optimization.get("recommendations", []):
            bullets.append(f"{rec['action']}: {rec['suggestion']}")
        
        if not bullets:
            bullets = [
                "Portfolio meets target return",
                "Risk within acceptable limits",
                "No immediate rebalancing needed"
            ]
        
        return self.build_message_response(
            reply=reply.strip(),
            bullets=bullets,
            conversation_id=conversation_id,
            model_info=self.get_model_info(),
            latency_ms=0
        )


# Singleton instance
calculation_agent = CalculationAgent()
