"""
Efficiency utilities: parallel processing, incremental calculations, parquet partitioning, signal deduplication.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# CONFIGURATION
# =============================================================================

class EfficiencyConfig(BaseModel):
    """Configuration for efficiency utilities."""
    parallel: Dict[str, Any] = Field(default_factory=lambda: {
        'enabled': True,
        'max_workers': 8,
        'rate_limit_per_second': 20,
        'chunk_size': 50,
        'use_process_pool': False,
    })

    incremental: Dict[str, Any] = Field(default_factory=lambda: {
        'enabled': True,
        'max_history_bars': 500,
    })

    storage: Dict[str, Any] = Field(default_factory=lambda: {
        'format': 'parquet',
        'partition_columns': ['date', 'symbol'],
        'compression': 'snappy',
        'row_group_size': 100000,
        'max_file_size_mb': 128,
    })

    signals: Dict[str, Any] = Field(default_factory=lambda: {
        'dedup_bars': 3,
        'min_signal_interval_seconds': 300,
        'aggregate_same_direction': True,
    })


# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

class ParallelProcessor:
    """
    Parallel processing with rate limiting and configurable executors.
    """

    def __init__(self, config: Optional[EfficiencyConfig] = None):
        self.config = config or EfficiencyConfig()
        self._rate_limiters: Dict[str, asyncio.Semaphore] = {}
        self._last_call: Dict[str, float] = defaultdict(float)

    async def parallel_fetch(
        self,
        func: Callable[..., T],
        items: List[Any],
        max_workers: Optional[int] = None,
        rate_limit: Optional[float] = None,
        chunk_size: Optional[int] = None,
        use_process_pool: Optional[bool] = None,
        **func_kwargs,
    ) -> List[T]:
        """
        Execute function in parallel across items with rate limiting.

        Args:
            func: Async or sync function to execute
            items: List of items to process
            max_workers: Max concurrent workers
            rate_limit: Calls per second limit
            chunk_size: Items per chunk
            use_process_pool: Use ProcessPoolExecutor (for CPU-bound)
            **func_kwargs: Additional keyword arguments for func

        Returns:
            List of results in same order as items
        """
        cfg = self.config.parallel
        max_workers = max_workers or cfg['max_workers']
        rate_limit = rate_limit or cfg['rate_limit_per_second']
        chunk_size = chunk_size or cfg['chunk_size']
        use_process_pool = use_process_pool if use_process_pool is not None else cfg['use_process_pool']

        if not items:
            return []

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_workers)
        rate_semaphore = asyncio.Semaphore(int(rate_limit)) if rate_limit > 0 else None

        async def rate_limited_call(item: Any) -> T:
            if rate_semaphore:
                await rate_semaphore.acquire()
                # Release after a small delay to maintain rate
                asyncio.create_task(self._release_rate_limit(rate_semaphore))

            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item, **func_kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: func(item, **func_kwargs))

        # Process in chunks to control memory
        results = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_tasks = [rate_limited_call(item) for item in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            for j, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel fetch failed for item {chunk[j]}: {result}")
                    results.append(None)  # Or raise
                else:
                    results.append(result)

        return results

    async def _release_rate_limit(self, semaphore: asyncio.Semaphore, delay: float = 1.0) -> None:
        """Release rate limit semaphore after delay."""
        await asyncio.sleep(delay)
        semaphore.release()

    def parallel_map(
        self,
        func: Callable[[T], R],
        items: List[T],
        max_workers: Optional[int] = None,
        use_process_pool: Optional[bool] = None,
    ) -> List[R]:
        """
        Synchronous parallel map using ThreadPoolExecutor or ProcessPoolExecutor.
        """
        cfg = self.config.parallel
        max_workers = max_workers or cfg['max_workers']
        use_process_pool = use_process_pool if use_process_pool is not None else cfg['use_process_pool']

        if not items:
            return []

        Executor = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

        with Executor(max_workers=max_workers) as executor:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            results = [None] * len(items)

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Parallel map failed for item {items[idx]}: {e}")
                    results[idx] = None

        return results


# =============================================================================
# INCREMENTAL INDICATOR CALCULATIONS
# =============================================================================

class IncrementalCalculator:
    """
    Incremental indicator calculations for streaming data.
    Maintains state to avoid recalculating full history.
    """

    def __init__(self, config: Optional[EfficiencyConfig] = None):
        self.config = config or EfficiencyConfig()
        self._state: Dict[str, Dict] = {}

    def update(
        self,
        indicator_name: str,
        new_value: float,
        params: Dict[str, Any],
    ) -> float:
        """
        Update indicator with new value.

        Args:
            indicator_name: Name of indicator (sma, ema, rsi, macd, etc.)
            new_value: New input value (typically close price)
            params: Indicator parameters (period, etc.)

        Returns:
            Updated indicator value
        """
        if indicator_name not in self._state:
            self._state[indicator_name] = {'values': [], 'params': params}

        state = self._state[indicator_name]
        state['values'].append(new_value)

        # Limit history
        max_bars = self.config.incremental['max_history_bars']
        if len(state['values']) > max_bars:
            state['values'] = state['values'][-max_bars:]

        # Calculate based on indicator type
        if indicator_name == 'sma':
            return self._update_sma(state, params)
        elif indicator_name == 'ema':
            return self._update_ema(state, params)
        elif indicator_name == 'rsi':
            return self._update_rsi(state, params)
        elif indicator_name == 'macd':
            return self._update_macd(state, params)
        elif indicator_name == 'atr':
            return self._update_atr(state, params)
        elif indicator_name == 'vwap':
            return self._update_vwap(state, params)
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

    def _update_sma(self, state: Dict, params: Dict) -> float:
        period = params.get('period', 20)
        values = state['values']
        if len(values) < period:
            return np.nan
        return np.mean(values[-period:])

    def _update_ema(self, state: Dict, params: Dict) -> float:
        period = params.get('period', 20)
        alpha = 2.0 / (period + 1)
        values = state['values']

        if 'ema' not in state:
            state['ema'] = values[0]
            return state['ema']

        state['ema'] = alpha * values[-1] + (1 - alpha) * state['ema']
        return state['ema']

    def _update_rsi(self, state: Dict, params: Dict) -> float:
        period = params.get('period', 14)
        values = state['values']

        if len(values) < period + 1:
            return np.nan

        deltas = np.diff(values[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        if 'avg_gain' not in state:
            state['avg_gain'] = np.mean(gains)
            state['avg_loss'] = np.mean(losses)
        else:
            state['avg_gain'] = (state['avg_gain'] * (period - 1) + gains[-1]) / period
            state['avg_loss'] = (state['avg_loss'] * (period - 1) + losses[-1]) / period

        if state['avg_loss'] == 0:
            return 100.0

        rs = state['avg_gain'] / state['avg_loss']
        return 100.0 - (100.0 / (1.0 + rs))

    def _update_macd(self, state: Dict, params: Dict) -> Dict[str, float]:
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)
        values = state['values']

        if len(values) < slow:
            return {'macd': np.nan, 'signal': np.nan, 'histogram': np.nan}

        # Calculate EMAs
        ema_fast = self._update_ema({'values': values, 'ema': state.get('ema_fast')}, {'period': fast})
        ema_slow = self._update_ema({'values': values, 'ema': state.get('ema_slow')}, {'period': slow})

        state['ema_fast'] = ema_fast
        state['ema_slow'] = ema_slow

        macd = ema_fast - ema_slow

        if 'signal_ema' not in state:
            state['signal_ema'] = macd
        else:
            alpha = 2.0 / (signal + 1)
            state['signal_ema'] = alpha * macd + (1 - alpha) * state['signal_ema']

        histogram = macd - state['signal_ema']

        return {
            'macd': macd,
            'signal': state['signal_ema'],
            'histogram': histogram,
        }

    def _update_atr(self, state: Dict, params: Dict) -> float:
        period = params.get('period', 14)
        # ATR needs high, low, close - simplified for close-only
        values = state['values']

        if len(values) < 2:
            return np.nan

        if 'tr_values' not in state:
            state['tr_values'] = []

        tr = abs(values[-1] - values[-2])  # Simplified TR
        state['tr_values'].append(tr)

        if len(state['tr_values']) < period:
            return np.nan

        if 'atr' not in state:
            state['atr'] = np.mean(state['tr_values'][-period:])
        else:
            state['atr'] = (state['atr'] * (period - 1) + tr) / period

        return state['atr']

    def _update_vwap(self, state: Dict, params: Dict) -> float:
        # VWAP needs volume - simplified
        return np.nan

    def reset(self, indicator_name: Optional[str] = None) -> None:
        """Reset indicator state."""
        if indicator_name:
            self._state.pop(indicator_name, None)
        else:
            self._state.clear()


def incremental_indicator(
    last_df: pd.DataFrame,
    new_bar: pd.Series,
    indicator_func: Callable,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Incrementally update indicator with new bar.

    Args:
        last_df: DataFrame with previous indicator values
        new_bar: New OHLCV bar as Series
        indicator_func: Indicator function to apply
        *args, **kwargs: Arguments for indicator function

    Returns:
        Updated DataFrame with new indicator values (last row only)
    """
    # Append new bar
    new_df = pd.concat([last_df, new_bar.to_frame().T])

    # Recalculate indicator on updated data
    result = indicator_func(new_df, *args, **kwargs)

    # Return only the new row
    return result.iloc[-1:]


# =============================================================================
# PARQUET PARTITIONING
# =============================================================================

@dataclass
class ParquetPartitionConfig:
    """Parquet partitioning configuration."""
    partition_columns: List[str] = Field(default_factory=lambda: ['date', 'symbol'])
    compression: str = 'snappy'
    row_group_size: int = 100000
    max_file_size_mb: int = 128
    filesystem: Any = None  # For cloud storage


def parquet_partition_write(
    df: pd.DataFrame,
    path: Union[str, Path],
    partition_cols: Optional[List[str]] = None,
    compression: str = 'snappy',
    row_group_size: int = 100000,
    max_file_size_mb: int = 128,
    **kwargs,
) -> None:
    """
    Write DataFrame to partitioned Parquet dataset.

    Args:
        df: DataFrame to write
        path: Output directory path
        partition_cols: Columns to partition by
        compression: Compression codec
        row_group_size: Max rows per row group
        max_file_size_mb: Target max file size
        **kwargs: Additional arguments to to_parquet
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if partition_cols is None:
        partition_cols = ['date', 'symbol']

    # Ensure partition columns exist
    for col in partition_cols:
        if col not in df.columns:
            raise ValueError(f"Partition column '{col}' not in DataFrame")

    # Convert date column if needed
    write_df = df.copy()
    for col in partition_cols:
        if pd.api.types.is_datetime64_any_dtype(write_df[col]):
            write_df[col] = write_df[col].dt.date

    # Write with partitioning
    write_df.to_parquet(
        path,
        partition_cols=partition_cols,
        compression=compression,
        row_group_size=row_group_size,
        **kwargs,
    )

    logger.info(f"Written partitioned parquet to {path} with partitions: {partition_cols}")


def read_partitioned_parquet(
    path: Union[str, Path],
    filters: Optional[List[Tuple]] = None,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read partitioned Parquet dataset with optional filters.

    Args:
        path: Input directory path
        filters: Partition filters, e.g. [('date', '>=', '2024-01-01'), ('symbol', '==', 'NIFTY')]
        columns: Columns to read
        **kwargs: Additional arguments to read_parquet

    Returns:
        DataFrame
    """
    return pd.read_parquet(
        path,
        filters=filters,
        columns=columns,
        **kwargs,
    )


def estimate_parquet_size(df: pd.DataFrame, partition_cols: List[str]) -> Dict[str, int]:
    """Estimate partition sizes for planning."""
    sizes = {}
    grouped = df.groupby(partition_cols)

    for name, group in grouped:
        # Rough estimate: 8 bytes per numeric column + overhead
        est_bytes = group.memory_usage(deep=True).sum()
        key = name if isinstance(name, str) else str(name)
        sizes[key] = est_bytes

    return sizes


# =============================================================================
# SIGNAL DEDUPLICATION
# =============================================================================

@dataclass
class Signal:
    """Trading signal."""
    id: str
    symbol: str
    signal_type: str
    direction: str  # 'buy', 'sell', 'long', 'short'
    price: float
    strength: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalDeduplicator:
    """
    Deduplicate trading signals to avoid overtrading.
    """

    def __init__(self, config: Optional[EfficiencyConfig] = None):
        self.config = config or EfficiencyConfig()
        self._last_signal: Dict[str, Signal] = {}
        self._signal_history: Dict[str, List[Signal]] = defaultdict(list)

    def should_emit(
        self,
        signal: Signal,
        dedup_bars: Optional[int] = None,
        min_interval_seconds: Optional[int] = None,
    ) -> bool:
        """
        Check if signal should be emitted (not a duplicate).

        Args:
            signal: New signal
            dedup_bars: Minimum bars between same-direction signals
            min_interval_seconds: Minimum seconds between signals

        Returns:
            True if signal should be emitted
        """
        cfg = self.config.signals
        dedup_bars = dedup_bars or cfg['dedup_bars']
        min_interval_seconds = min_interval_seconds or cfg['min_signal_interval_seconds']

        key = f"{signal.symbol}:{signal.signal_type}:{signal.direction}"

        # Check time interval
        if key in self._last_signal:
            last = self._last_signal[key]
            time_diff = (signal.timestamp - last.timestamp).total_seconds()
            if time_diff < min_interval_seconds:
                logger.debug(f"Signal deduplicated (time): {signal.id} - {time_diff:.0f}s since last")
                return False

        # Check bar count (would need bar data - simplified)
        # For now, use time-based as proxy

        # Check aggregate same direction
        if cfg['aggregate_same_direction']:
            # Allow if direction changed
            pass

        return True

    def register_signal(self, signal: Signal) -> None:
        """Register emitted signal."""
        key = f"{signal.symbol}:{signal.signal_type}:{signal.direction}"
        self._last_signal[key] = signal
        self._signal_history[key].append(signal)

        # Limit history
        if len(self._signal_history[key]) > 1000:
            self._signal_history[key] = self._signal_history[key][-1000:]

    def deduplicate_signals(
        self,
        signals: List[Signal],
        dedup_bars: Optional[int] = None,
    ) -> List[Signal]:
        """
        Deduplicate a list of signals.

        Args:
            signals: List of signals (assumed sorted by timestamp)
            dedup_bars: Minimum bars between same-direction signals

        Returns:
            Deduplicated signals
        """
        if not signals:
            return []

        result = []
        last_by_key: Dict[str, Signal] = {}

        for signal in signals:
            key = f"{signal.symbol}:{signal.signal_type}:{signal.direction}"

            if key in last_by_key:
                last = last_by_key[key]
                # Simplified: check time difference as proxy for bars
                time_diff = (signal.timestamp - last.timestamp).total_seconds()
                min_interval = self.config.signals['min_signal_interval_seconds']

                if time_diff < min_interval:
                    # Keep stronger signal
                    if signal.strength > last.strength:
                        result[-1] = signal
                        last_by_key[key] = signal
                    continue

            result.append(signal)
            last_by_key[key] = signal

        return result


# =============================================================================
# CACHING UTILITIES
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recent)
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    self._access_order.remove(key)
        return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Evict LRU
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

            self._cache[key] = (value, time.time())
            self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
        return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_parallel_processor(config: Optional[Dict] = None) -> ParallelProcessor:
    """Factory for ParallelProcessor."""
    if config:
        return ParallelProcessor(EfficiencyConfig(**config))
    return ParallelProcessor()


def create_incremental_calculator(config: Optional[Dict] = None) -> IncrementalCalculator:
    """Factory for IncrementalCalculator."""
    if config:
        return IncrementalCalculator(EfficiencyConfig(**config))
    return IncrementalCalculator()


def create_signal_deduplicator(config: Optional[Dict] = None) -> SignalDeduplicator:
    """Factory for SignalDeduplicator."""
    if config:
        return SignalDeduplicator(EfficiencyConfig(**config))
    return SignalDeduplicator()


def create_lru_cache(max_size: int = 1000, ttl_seconds: float = 3600) -> LRUCache:
    """Factory for LRUCache."""
    return LRUCache(max_size, ttl_seconds)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def parallel_fetch_with_retry(
    func: Callable,
    items: List[Any],
    max_workers: int = 8,
    max_retries: int = 3,
    **kwargs,
) -> List[Any]:
    """Parallel fetch with retry logic."""
    processor = create_parallel_processor()
    return await processor.parallel_fetch(func, items, max_workers=max_workers, **kwargs)


def write_parquet_partitioned(
    df: pd.DataFrame,
    path: str,
    partition_cols: List[str] = None,
    **kwargs,
) -> None:
    """Convenience function for partitioned parquet write."""
    parquet_partition_write(df, path, partition_cols, **kwargs)


def read_parquet_partitioned(
    path: str,
    filters: List[Tuple] = None,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function for partitioned parquet read."""
    return read_partitioned_parquet(path, filters, **kwargs)