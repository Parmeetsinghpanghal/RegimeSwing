"""
News Fetcher Module - Fetches news and sentiment from free Indian financial sources.
"""

import feedparser
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sqlite3
import logging
import re
from bs4 import BeautifulSoup

# Optional: Transformers for FinBERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NewsArticle:
    """Container for news article data."""
    def __init__(self, title: str, summary: str, link: str, published: datetime, source: str):
        self.title = title
        self.summary = summary
        self.link = link
        self.published = published
        self.source = source


class NewsFetcher:
    """
    Fetch news from free Indian financial sources and analyze sentiment.
    """

    def __init__(self, config: dict):
        self.config = config
        self.news_config = config.get('news', {})
        self.cache_dir = Path("data/cache/news")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FinBERT
        self.sentiment_model = None
        self.tokenizer = None
        if self.news_config.get('sentiment_model') == 'finbert' and True:  # TRANSFORMERS_AVAILABLE
            self._init_finbert()

        # Initialize SQLite cache
        self._init_cache_db()

        # RSS feed URLs
        self.rss_feeds = {
            'moneycontrol_market': 'https://www.moneycontrol.com/rss/marketnews.xml',
            'moneycontrol_stocks': 'https://www.moneycontrol.com/rss/stockmarketnews.xml',
            'etmarkets': 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'etmarkets_news': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'livemint': 'https://www.livemint.com/rss/markets',
            'nse_announcements': 'https://www.nseindia.com/api/corporate-announcements',
        }

    def _init_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis."""
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_model.eval()
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e}")
            self.sentiment_model = None

    def _init_cache_db(self):
        """Initialize SQLite cache for news."""
        db_path = self.cache_dir / "news_cache.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    title TEXT,
                    summary TEXT,
                    link TEXT UNIQUE,
                    published TIMESTAMP,
                    symbols TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON news_cache(published)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols ON news_cache(symbols)")

    def _get_cache_db(self):
        return self.cache_dir / "news_cache.db"

    def fetch_rss(self, feed_url: str, source_name: str) -> List[Dict]:
        """Parse RSS feed."""
        try:
            feed = feedparser.parse(feed_url)
            entries = []
            for entry in feed.entries:
                entries.append({
                    'source': source_name,
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', entry.get('description', '')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', entry.get('updated', '')),
                })
            return entries
        except Exception as e:
            logger.warning(f"RSS fetch failed for {source_name}: {e}")
            return []

    def fetch_nse_announcements(self) -> List[Dict]:
        """Fetch NSE corporate announcements."""
        # NSE API requires headers - simplified version
        return []

    def fetch_all_sources(self) -> List[Dict]:
        """Fetch from all configured RSS sources."""
        all_entries = []
        for source, url in self.rss_feeds.items():
            if source != 'nse_announcements':
                entries = self.fetch_rss(url, source)
                all_entries.extend(entries)
                logger.info(f"Fetched {len(entries)} entries from {source}")
        return all_entries

    def extract_symbols(self, text: str, known_symbols: List[str]) -> List[str]:
        """Extract stock symbols from text."""
        found = []
        text_upper = text.upper()
        for symbol in known_symbols:
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, text_upper):
                found.append(symbol)
        return found

    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment using FinBERT.

        Returns:
            List of dicts with 'score' (-1 to 1) and 'label' ('positive', 'negative', 'neutral')
        """
        if not self.sentiment_model or not texts:
            return [{'score': 0.0, 'label': 'neutral'} for _ in texts]

        results = []
        batch_size = 8

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # FinBERT labels: 0=positive, 1=negative, 2=neutral
                for prob in probs:
                    pos, neg, neu = prob[0].item(), prob[1].item(), prob[2].item()
                    if pos > neg and pos > neu:
                        label = 'positive'
                        score = pos
                    elif neg > pos and neg > neu:
                        label = 'negative'
                        score = -neg
                    else:
                        label = 'neutral'
                        score = 0.0
                    results.append({'score': score, 'label': label})

            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                results.extend([{'score': 0.0, 'label': 'neutral'}] * len(batch))

        return results

    def get_cached_news(self, hours: int = 4) -> pd.DataFrame:
        """Get cached news within time window."""
        db_path = self.cache_dir / "news_cache.db"
        if not db_path.exists():
            return pd.DataFrame()

        cutoff = datetime.now() - timedelta(hours=hours)
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM news_cache WHERE published > ? ORDER BY published DESC",
                conn,
                params=[cutoff.isoformat()]
            )
        return df

    def cache_news(self, entries: List[Dict]):
        """Cache news entries to SQLite."""
        db_path = self.cache_dir / "news_cache.db"
        with sqlite3.connect(db_path) as conn:
            for entry in entries:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO news_cache
                        (source, title, summary, link, published, symbols, sentiment_score, sentiment_label)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.get('source'),
                        entry.get('title'),
                        entry.get('summary'),
                        entry.get('link'),
                        entry.get('published'),
                        ','.join(entry.get('symbols', [])),
                        entry.get('sentiment_score'),
                        entry.get('sentiment_label')
                    ))
                except Exception as e:
                    logger.warning(f"Cache insert failed: {e}")

    def fetch_and_process(self, known_symbols: List[str] = None) -> pd.DataFrame:
        """
        Main method: fetch, analyze, cache, return DataFrame.
        """
        if known_symbols is None:
            known_symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
                            "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "ASIANPAINT", "MARUTI", "SUNPHARMA",
                            "TITAN", "AXISBANK", "KOTAKBANK", "M&M", "ULTRACEMCO", "BAJAJFINSV"]

        # Check cache first
        cached = self.get_cached_news(hours=self.news_config.get('cache_hours', 4))
        if not cached.empty:
            logger.info(f"Returning {len(cached)} cached news entries")
            return cached

        # Fetch fresh
        logger.info("Fetching fresh news...")
        raw_entries = self.fetch_all_sources()

        if not raw_entries:
            return pd.DataFrame()

        # Process entries
        processed = []
        texts = []
        for entry in raw_entries:
            full_text = f"{entry['title']}. {entry['summary']}"
            texts.append(full_text)
            entry['full_text'] = full_text

        # Analyze sentiment
        sentiments = self.analyze_sentiment(texts)

        # Process each entry
        for entry, sent in zip(raw_entries, sentiments):
            symbols = self.extract_symbols(entry['full_text'], known_symbols) if known_symbols else []
            entry['symbols'] = symbols
            entry['sentiment_score'] = sent['score']
            entry['sentiment_label'] = sent['label']
            processed.append(entry)

        # Cache
        self.cache_news(processed)

        return pd.DataFrame(processed)

    def get_symbol_sentiment(self, symbol: str, hours: int = 24) -> Dict:
        """Get aggregated sentiment for a specific symbol."""
        df = self.get_cached_news(hours=hours)
        if df.empty:
            return {'score': 0.0, 'label': 'neutral', 'count': 0}

        # Filter for symbol
        sym_news = df[df['symbols'].str.contains(symbol, na=False)]
        if sym_news.empty:
            return {'score': 0.0, 'label': 'neutral', 'count': 0}

        avg_score = sym_news['sentiment_score'].mean()
        if avg_score > 0.1:
            label = 'positive'
        elif avg_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'score': float(avg_score),
            'label': label,
            'count': len(sym_news),
            'positive': len(sym_news[sym_news['sentiment_label'] == 'positive']),
            'negative': len(sym_news[sym_news['sentiment_label'] == 'negative']),
            'neutral': len(sym_news[sym_news['sentiment_label'] == 'neutral'])
        }

    def get_sector_sentiment(self, sector: str, hours: int = 24) -> Dict:
        """Get sentiment for sector (placeholder - would need sector mapping)."""
        return {'score': 0.0, 'label': 'neutral', 'count': 0}

    def is_high_uncertainty(self, symbol: str, date: datetime = None) -> bool:
        """Check if high uncertainty period for symbol (earnings, major news)."""
        if date is None:
            date = datetime.now()

        # Check earnings calendar (placeholder)
        # Check major negative news
        sentiment = self.get_symbol_sentiment(symbol, hours=48)
        threshold = self.news_config.get('avoid_threshold', -0.5)

        return sentiment['score'] < threshold

    def get_earnings_calendar(self, symbols: List[str], days_ahead: int = 30) -> Dict:
        """Get upcoming earnings dates (placeholder - would integrate with NSE/calendar API)."""
        return {}


def create_news_fetcher(config: dict) -> NewsFetcher:
    return NewsFetcher(config)


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    fetcher = create_news_fetcher(config)

    # Test fetch
    news_df = fetcher.fetch_and_process(['RELIANCE', 'TCS', 'HDFCBANK'])
    print(f"Fetched {len(news_df)} news entries")

    if not news_df.empty:
        print(news_df[['source', 'title', 'symbols', 'sentiment_label', 'sentiment_score']].head(10))

    # Test symbol sentiment
    sent = fetcher.get_symbol_sentiment('RELIANCE')
    print(f"\nRELIANCE sentiment: {sent}")