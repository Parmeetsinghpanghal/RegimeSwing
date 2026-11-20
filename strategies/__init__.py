"""
Strategies Package - Swing and Options trading strategies.
"""

from .swing import (
    run_all_strategies,
    get_all_strategies,
    TrendPullbackStrategy,
    BreakoutVolumeStrategy,
    MeanReversionBBStrategy,
)

from .options import (
    run_all_options_strategies,
    get_all_options_strategies,
    CashSecuredPutsStrategy,
    CoveredCallsStrategy,
    WheelStrategy,
    CreditSpreadsStrategy,
)

from .scoring import (
    MultiFactorScorer,
    SwingScore,
    FactorScore,
    create_scorer,
    score_and_rank,
)

__all__ = [
    # Swing strategies
    'run_all_strategies',
    'get_all_strategies',
    'TrendPullbackStrategy',
    'BreakoutVolumeStrategy',
    'MeanReversionBBStrategy',

    # Options strategies
    'run_all_options_strategies',
    'get_all_options_strategies',
    'CashSecuredPutsStrategy',
    'CoveredCallsStrategy',
    'WheelStrategy',
    'CreditSpreadsStrategy',

    # Scoring
    'MultiFactorScorer',
    'SwingScore',
    'FactorScore',
    'create_scorer',
    'score_and_rank',
]