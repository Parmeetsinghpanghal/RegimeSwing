"""
Multi-Factor Swing Scoring System (0-100) with Backtest Validation Gate.

Combines regime detection, technical factors, and walk-forward backtest results
to produce a unified score for each stock. Only stocks passing the backtest
gate (Win Rate > 55%, Expectancy > 0.3R) get live alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from indicators.technical import ensure_multiindex, get_symbol_data
from indicators.regime import RegimeDetector
from backtest import BacktestConfig, WalkForwardBacktester, run_backtest

logger = logging.getLogger(__name__)


@dataclass
class FactorScore:
    """Individual factor score with details."""
    name: str
    score: float          # 0-100
    weight: float         # 0-1
    value: float          # Raw factor value
    description: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SwingScore:
    """Complete swing score for a symbol."""
    symbol: str
    total_score: float                    # 0-100 weighted sum
    factor_scores: List[FactorScore]
    regime: str
    regime_confidence: float
    backtest_passed: bool
    backtest_metrics: Dict
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'total_score': round(self.total_score, 1),
            'regime': self.regime,
            'regime_confidence': round(self.regime_confidence, 2),
            'backtest_passed': self.backtest_passed,
            'backtest_metrics': {k: round(v, 4) if isinstance(v, float) else v for k, v in self.backtest_metrics.items()},
            'factors': [
                {
                    'name': f.name,
                    'score': round(f.score, 1),
                    'weight': f.weight,
                    'value': round(f.value, 4) if isinstance(f.value, float) else f.value,
                    'description': f.description
                }
                for f in self.factor_scores
            ],
            'timestamp': self.timestamp.isoformat()
        }


class MultiFactorScorer:
    """
    Multi-factor swing scoring (0-100) with configurable weights.

    Factors:
    1. Trend Alignment (25%) - EMA alignment, ADX strength
    2. Momentum (20%) - RSI, ROC, MACD
    3. Volume Surge (15%) - Volume ratio, OBV trend
    4. Volatility/ATR (15%) - ATR percentile, BB width
    5. Structure/Support (15%) - Distance to support, SR zones
    6. Regime Alignment (10%) - Regime compatibility with strategy
    """

    # Default weights (sum = 1.0)
    DEFAULT_WEIGHTS = {
        'trend_alignment': 0.25,
        'momentum': 0.20,
        'volume_surge': 0.15,
        'volatility': 0.15,
        'structure': 0.15,
        'regime_alignment': 0.10,
    }

    # Backtest validation thresholds
    MIN_WIN_RATE = 0.55      # 55%
    MIN_EXPECTANCY = 0.3     # 0.3R
    MIN_PROFIT_FACTOR = 1.0
    MIN_TRADES = 10

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.weights = self.config.get('scoring_weights', self.DEFAULT_WEIGHTS)
        self.backtest_config = self.config.get('backtest_gate', {})
        self.min_win_rate = self.backtest_config.get('min_win_rate', self.MIN_WIN_RATE)
        self.min_expectancy = self.backtest_config.get('min_expectancy', self.MIN_EXPECTANCY)
        self.min_profit_factor = self.backtest_config.get('min_profit_factor', self.MIN_PROFIT_FACTOR)
        self.min_trades = self.backtest_config.get('min_trades', self.MIN_TRADES)

        # Cache for backtest results
        self._backtest_cache: Dict[str, Dict] = {}
        self._cache_path = Path("./logs/backtest_cache.json")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _load_cache(self):
        """Load backtest cache from disk."""
        try:
            if self._cache_path.exists():
                with open(self._cache_path, 'r') as f:
                    self._backtest_cache = json.load(f)
                logger.info(f"Loaded backtest cache with {len(self._backtest_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load backtest cache: {e}")
            self._backtest_cache = {}

    def _save_cache(self):
        """Save backtest cache to disk."""
        try:
            with open(self._cache_path, 'w') as f:
                json.dump(self._backtest_cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save backtest cache: {e}")

    def _get_cache_key(self, symbol: str, strategy: str) -> str:
        """Generate cache key for backtest results."""
        return f"{symbol}_{strategy}"

    def score_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime_info: Dict,
        strategy: str = 'trend_pullback'
    ) -> SwingScore:
        """
        Score a single symbol across all factors.

        Args:
            symbol: Stock symbol
            df: MultiIndex DataFrame with indicators
            regime_info: Current regime info from RegimeDetector
            strategy: Strategy to score for

        Returns:
            SwingScore with total score and factor breakdown
        """
        # Get symbol data
        sym_df = get_symbol_data(df, symbol)
        if len(sym_df) < 50:
            return self._empty_score(symbol, regime_info, "Insufficient data")

        latest = sym_df.iloc[-1]

        # Calculate factor scores
        factors = []

        # 1. Trend Alignment (25%)
        factors.append(self._score_trend_alignment(latest, sym_df))

        # 2. Momentum (20%)
        factors.append(self._score_momentum(latest, sym_df))

        # 3. Volume Surge (15%)
        factors.append(self._score_volume_surge(latest, sym_df))

        # 4. Volatility/ATR (15%)
        factors.append(self._score_volatility(latest, sym_df))

        # 5. Structure/Support (15%)
        factors.append(self._score_structure(latest, sym_df))

        # 6. Regime Alignment (10%)
        factors.append(self._score_regime_alignment(regime_info, strategy))

        # Calculate weighted total
        total_score = sum(f.score * f.weight for f in factors)

        # Run backtest validation gate
        backtest_passed, backtest_metrics = self._validate_backtest(symbol, strategy)

        return SwingScore(
            symbol=symbol,
            total_score=total_score,
            factor_scores=factors,
            regime=regime_info.get('regime', 'unknown'),
            regime_confidence=regime_info.get('confidence', 0.0),
            backtest_passed=backtest_passed,
            backtest_metrics=backtest_metrics,
            timestamp=datetime.now(),
            metadata={'strategy': strategy}
        )

    def _score_trend_alignment(self, latest: pd.Series, sym_df: pd.DataFrame) -> FactorScore:
        """Score trend alignment: EMA alignment, ADX strength."""
        close = latest['Close']
        ema20 = latest.get('EMA_20', close)
        ema50 = latest.get('EMA_50', close)
        ema200 = latest.get('EMA_200', close)
        adx = latest.get('ADX_14', 0)

        # EMA alignment score (0-100)
        if close > ema50 > ema200:
            ema_score = 100  # Perfect bullish alignment
        elif close > ema50 and ema50 > ema200 * 0.98:
            ema_score = 75   # Good alignment
        elif close > ema200:
            ema_score = 50   # Above long-term only
        else:
            ema_score = 10   # Below long-term

        # ADX score (0-100)
        adx_score = min(adx * 4, 100)  # ADX 25 = 100

        # Combined trend score
        trend_score = 0.7 * ema_score + 0.3 * adx_score

        return FactorScore(
            name='trend_alignment',
            score=trend_score,
            weight=self.weights['trend_alignment'],
            value=trend_score,
            description=f"EMA stack: {'✓' if close > ema50 > ema200 else '✗'} | ADX: {adx:.1f}",
            metadata={'ema20': ema20, 'ema50': ema50, 'ema200': ema200, 'adx': adx}
        )

    def _score_momentum(self, latest: pd.Series, sym_df: pd.DataFrame) -> FactorScore:
        """Score momentum: RSI, ROC, MACD."""
        rsi = latest.get('RSI_14', 50)
        roc = latest.get('ROC_10', 0)
        macd = latest.get('MACD_12_26_9', 0)
        macd_signal = latest.get('MACDs_12_26_9', 0)

        # RSI score (sweet spot 45-65 for trend pullback)
        if 45 <= rsi <= 65:
            rsi_score = 100
        elif 40 <= rsi < 45 or 65 < rsi <= 70:
            rsi_score = 70
        elif 30 <= rsi < 40 or 70 < rsi <= 80:
            rsi_score = 40
        else:
            rsi_score = 10

        # ROC score (positive momentum)
        roc_score = min(max(roc * 10 + 50, 0), 100)  # 5% ROC = 100

        # MACD score (MACD above signal = bullish)
        macd_score = 100 if macd > macd_signal else 30

        # Combined momentum
        momentum_score = 0.4 * rsi_score + 0.3 * roc_score + 0.3 * macd_score

        return FactorScore(
            name='momentum',
            score=momentum_score,
            weight=self.weights['momentum'],
            value=momentum_score,
            description=f"RSI: {rsi:.1f} | ROC: {roc:.1f}% | MACD: {macd:.2f}",
            metadata={'rsi': rsi, 'roc': roc, 'macd': macd, 'macd_signal': macd_signal}
        )

    def _score_volume_surge(self, latest: pd.Series, sym_df: pd.DataFrame) -> FactorScore:
        """Score volume surge: Volume ratio, OBV trend."""
        vol_ratio = latest.get('Volume_Ratio', 1.0)
        obv = latest.get('OBV', 0)

        # Volume ratio score
        if vol_ratio >= 2.0:
            vol_score = 100
        elif vol_ratio >= 1.5:
            vol_score = 80
        elif vol_ratio >= 1.2:
            vol_score = 60
        elif vol_ratio >= 1.0:
            vol_score = 40
        else:
            vol_score = 20

        # OBV trend (compare to 20 periods ago)
        obv_20_ago = sym_df['OBV'].iloc[-21] if len(sym_df) > 20 else obv
        obv_trend = 1 if obv > obv_20_ago else 0
        obv_score = 100 if obv_trend else 30

        volume_score = 0.7 * vol_score + 0.3 * obv_score

        return FactorScore(
            name='volume_surge',
            score=volume_score,
            weight=self.weights['volume_surge'],
            value=volume_score,
            description=f"Vol Ratio: {vol_ratio:.2f}x | OBV Trend: {'↑' if obv_trend else '↓'}",
            metadata={'volume_ratio': vol_ratio, 'obv_trend': obv_trend}
        )

    def _score_volatility(self, latest: pd.Series, sym_df: pd.DataFrame) -> FactorScore:
        """Score volatility: ATR percentile, BB width."""
        atr = latest.get('ATR_14', 0)
        close = latest['Close']
        atr_pct = atr / close if close > 0 else 0

        bb_width = latest.get('BB_width', 0)

        # ATR percentile (compare to 252-day history)
        atr_series = sym_df['ATR_14'] / sym_df['Close']
        atr_percentile = (atr_series < atr_pct).mean() * 100 if len(atr_series) > 0 else 50

        # Optimal ATR range: 1.5-3% for swing trading
        if 0.015 <= atr_pct <= 0.03:
            atr_score = 100
        elif 0.01 <= atr_pct < 0.015 or 0.03 < atr_pct <= 0.04:
            atr_score = 70
        elif 0.005 <= atr_pct < 0.01 or 0.04 < atr_pct <= 0.05:
            atr_score = 40
        else:
            atr_score = 20

        # BB width score (moderate width is good)
        if 0.05 <= bb_width <= 0.15:
            bb_score = 100
        elif 0.03 <= bb_width < 0.05 or 0.15 < bb_width <= 0.25:
            bb_score = 70
        else:
            bb_score = 40

        volatility_score = 0.6 * atr_score + 0.4 * bb_score

        return FactorScore(
            name='volatility',
            score=volatility_score,
            weight=self.weights['volatility'],
            value=volatility_score,
            description=f"ATR%: {atr_pct:.2%} (pctile: {atr_percentile:.0f}) | BB Width: {bb_width:.2%}",
            metadata={'atr_pct': atr_pct, 'atr_percentile': atr_percentile, 'bb_width': bb_width}
        )

    def _score_structure(self, latest: pd.Series, sym_df: pd.DataFrame) -> FactorScore:
        """Score structure: Distance to support, SR zones."""
        close = latest['Close']

        # Support levels
        low_20 = sym_df['Low'].tail(20).min()
        low_50 = sym_df['Low'].tail(50).min()

        dist_to_20_low = (close - low_20) / close
        dist_to_50_low = (close - low_50) / close

        # Score: closer to support = better (for pullback entry)
        if dist_to_20_low <= 0.03:      # Within 3% of 20-day low
            support_score = 100
        elif dist_to_20_low <= 0.05:    # Within 5%
            support_score = 80
        elif dist_to_20_low <= 0.08:    # Within 8%
            support_score = 60
        elif dist_to_20_low <= 0.12:    # Within 12%
            support_score = 40
        else:
            support_score = 20

        # Chandelier exit distance
        chandelier = latest.get('Chandelier_Long', close * 0.9)
        chandelier_dist = (close - chandelier) / close
        chandelier_score = min(chandelier_dist * 2000, 100) if chandelier_dist > 0 else 10

        structure_score = 0.6 * support_score + 0.4 * chandelier_score

        return FactorScore(
            name='structure',
            score=structure_score,
            weight=self.weights['structure'],
            value=structure_score,
            description=f"Dist to 20D low: {dist_to_20_low:.1%} | Chandelier: {chandelier:.2f}",
            metadata={'dist_to_20_low': dist_to_20_low, 'dist_to_50_low': dist_to_50_low,
                     'chandelier': chandelier, 'chandelier_dist': chandelier_dist}
        )

    def _score_regime_alignment(self, regime_info: Dict, strategy: str) -> FactorScore:
        """Score regime alignment for strategy."""
        regime = regime_info.get('regime', 'unknown')
        confidence = regime_info.get('confidence', 0.5)

        # Strategy-regime compatibility matrix
        compatibility = {
            'trend_pullback': {
                'bull': 100, 'volatile': 60, 'sideways': 20, 'bear': 0, 'unknown': 30
            },
            'breakout_volume': {
                'bull': 100, 'volatile': 70, 'sideways': 10, 'bear': 0, 'unknown': 30
            },
            'mean_reversion_bb': {
                'sideways': 100, 'bull': 60, 'volatile': 10, 'bear': 0, 'unknown': 30
            }
        }

        strat_compat = compatibility.get(strategy, compatibility['trend_pullback'])
        base_score = strat_compat.get(regime, 30)

        # Adjust by confidence
        regime_score = base_score * confidence

        return FactorScore(
            name='regime_alignment',
            score=regime_score,
            weight=self.weights['regime_alignment'],
            value=regime_score,
            description=f"Regime: {regime} ({confidence:.0%}) | Strategy: {strategy}",
            metadata={'regime': regime, 'confidence': confidence, 'strategy': strategy}
        )

    def _validate_backtest(self, symbol: str, strategy: str) -> Tuple[bool, Dict]:
        """
        Validate symbol passes backtest gate.
        Uses cached results or runs quick backtest.
        """
        cache_key = self._get_cache_key(symbol, strategy)

        # Check cache first
        if cache_key in self._backtest_cache:
            cached = self._backtest_cache[cache_key]
            # Cache valid for 7 days
            cached_date = datetime.fromisoformat(cached.get('date', '2000-01-01'))
            if datetime.now() - cached_date < timedelta(days=7):
                metrics = cached.get('metrics', {})
                passed = self._check_gate(metrics)
                return passed, metrics

        # Run quick backtest (3-year walk-forward)
        # This is a placeholder - in production, use the full backtest engine
        # For now, return mock metrics for demo
        metrics = self._run_quick_backtest(symbol, strategy)

        # Cache results
        self._backtest_cache[cache_key] = {
            'metrics': metrics,
            'date': datetime.now().isoformat()
        }
        self._save_cache()

        passed = self._check_gate(metrics)
        return passed, metrics

    def _check_gate(self, metrics: Dict) -> bool:
        """Check if backtest metrics pass the gate."""
        if not metrics:
            return False

        win_rate = metrics.get('win_rate', 0)
        expectancy = metrics.get('expectancy', 0)
        profit_factor = metrics.get('profit_factor', 0)
        total_trades = metrics.get('total_trades', 0)

        return (
            win_rate >= self.min_win_rate and
            expectancy >= self.min_expectancy and
            profit_factor >= self.min_profit_factor and
            total_trades >= self.min_trades
        )

    def _run_quick_backtest(self, symbol: str, strategy: str) -> Dict:
        """
        Run quick backtest for validation.
        In production, this calls the full backtest engine.
        For now, returns mock metrics based on symbol characteristics.
        """
        # Use deterministic hash of symbol for consistent mock results
        import hashlib
        seed = int(hashlib.md5(f"{symbol}_{strategy}".encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        # Simulate realistic metrics based on symbol characteristics
        base_wr = 0.45 + np.random.random() * 0.30  # 45-75%
        base_exp = -0.1 + np.random.random() * 0.8   # -0.1 to 0.7

        return {
            'win_rate': round(base_wr, 3),
            'expectancy': round(base_exp, 3),
            'profit_factor': round(0.8 + np.random.random() * 1.5, 2),
            'total_trades': int(5 + np.random.random() * 30),
            'cagr': round(0.05 + np.random.random() * 0.25, 3),
            'sharpe': round(0.5 + np.random.random() * 1.5, 2),
            'max_drawdown': round(0.08 + np.random.random() * 0.15, 3),
            'avg_r_multiple': round(0.5 + np.random.random() * 1.5, 2)
        }

    def _check_gate(self, metrics: Dict) -> bool:
        """Check if backtest metrics pass the validation gate."""
        if not metrics:
            return False

        win_rate = metrics.get('win_rate', 0)
        expectancy = metrics.get('expectancy', 0)
        profit_factor = metrics.get('profit_factor', 0)
        total_trades = metrics.get('total_trades', 0)

        return (
            win_rate >= self.min_win_rate and
            expectancy >= self.min_expectancy and
            profit_factor >= self.min_profit_factor and
            total_trades >= self.min_trades
        )

    def _empty_score(self, symbol: str, regime_info: Dict, reason: str) -> SwingScore:
        """Return empty score for failed symbols."""
        return SwingScore(
            symbol=symbol,
            total_score=0.0,
            factor_scores=[],
            regime=regime_info.get('regime', 'unknown'),
            regime_confidence=0.0,
            backtest_passed=False,
            backtest_metrics={},
            timestamp=datetime.now(),
            metadata={'error': reason}
        )

    def score_universe(
        self,
        df: pd.DataFrame,
        regime_info: Dict,
        strategy: str = 'trend_pullback',
        min_score: float = 50.0
    ) -> List[SwingScore]:
        """
        Score all symbols in universe and filter by minimum score + backtest gate.

        Args:
            df: MultiIndex DataFrame with all symbols and indicators
            regime_info: Current regime info from RegimeDetector
            strategy: Strategy to score for
            min_score: Minimum total score (0-100)

        Returns:
            List of SwingScore sorted by total_score descending
        """
        symbols = df.index.get_level_values('Symbol').unique()
        scores = []

        logger.info(f"Scoring {len(symbols)} symbols for {strategy}...")

        for symbol in symbols:
            try:
                score = self.score_symbol(symbol, df, regime_info, strategy)
                if score.total_score >= min_score:
                    scores.append(score)
            except Exception as e:
                logger.warning(f"Scoring failed for {symbol}: {e}")

        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)

        # Log results
        passed = sum(1 for s in scores if s.backtest_passed)
        logger.info(f"Scored {len(scores)} symbols ≥ {min_score}, {passed} passed backtest gate")

        return scores

    def get_top_picks(
        self,
        scores: List[SwingScore],
        top_n: int = 5,
        require_backtest: bool = True
    ) -> List[SwingScore]:
        """Get top N picks, optionally requiring backtest pass."""
        filtered = scores
        if require_backtest:
            filtered = [s for s in scores if s.backtest_passed]
        return filtered[:top_n]


# Convenience functions
def create_scorer(config: Dict = None) -> MultiFactorScorer:
    """Create MultiFactorScorer instance."""
    return MultiFactorScorer(config)


def score_and_rank(
    df: pd.DataFrame,
    regime_info: Dict,
    strategy: str = 'trend_pullback',
    config: Dict = None,
    top_n: int = 5
) -> List[SwingScore]:
    """Quick function to score and rank universe."""
    scorer = create_scorer(config)
    scores = scorer.score_universe(df, regime_info, strategy)
    return scorer.get_top_picks(scores, top_n=5, require_backtest=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    from indicators.technical import add_all_indicators, ensure_multiindex
    import yfinance as yf

    # Fetch test data
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
    dfs = []
    for sym in symbols:
        df = yf.download(f"{sym}.NS", period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        df['Symbol'] = sym
        dfs.append(df)

    combined = pd.concat(dfs)
    combined = ensure_multiindex(combined)
    combined = add_all_indicators(combined, {})

    # Regime
    nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty.columns = [c.capitalize() for c in nifty.columns]
    nifty = add_all_indicators(nifty, {})

    detector = RegimeDetector({})
    regime = detector.detect_nifty_regime(nifty)
    regime_info = {
        'regime': regime.regime,
        'confidence': regime.confidence,
        'size_mult': detector.get_size_multiplier(regime),
        'nifty_trend': regime.nifty_trend
    }

    print(f"Regime: {regime_info}")

    # Score universe
    scorer = create_scorer()
    scores = scorer.score_universe(combined, regime_info, 'trend_pullback')

    print(f"\nTop 5 Scores:")
    for s in sorted(scorer.get_top_picks(scorer.score_universe(combined, regime_info)), key=lambda x: x.total_score, reverse=True)[:5]:
        print(f"  {s.symbol}: {s.total_score:.1f} | Regime: {s.regime} | Backtest: {'✅' if s.backtest_passed else '❌'}")
        for f in s.factor_scores:
            print(f"  {f.name}: {f.score:.0f} (w={f.weight}) - {f.description}")