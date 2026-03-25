"""
PoliticalFilter – Rule-based headline classification gate.
==========================================================
Two-gate system that filters headlines before they reach the ML model:
  Gate 1: Non-political content → immediately classified as Neutral
  Gate 2: Political but unbiased → classified as Neutral
  Gate 3: Politically biased → forwarded to BERT for Left/Right/Neutral
"""

import re
from enum import Enum
from typing import List


class FilterResult(Enum):
    """Outcome of rule-based filtering."""
    NON_POLITICAL = "non_political"
    NEUTRAL_POLITICAL = "neutral_political"
    BIASED_POLITICAL = "biased_political"


class PoliticalFilter:
    """
    Word-boundary-aware keyword filter for Indian news headlines.

    Usage:
        pf = PoliticalFilter()
        result = pf.classify("Hyderabad weather forecast for tomorrow")
        # → FilterResult.NON_POLITICAL
    """

    # ── Non-political topics (Gate 1) ────────────────────────
    NON_POLITICAL_KEYWORDS: List[str] = [
        # Weather & environment
        "weather", "temperature", "rain", "rainfall", "humidity",
        "forecast", "heatwave", "cold wave", "monsoon", "cyclone",
        # Sports
        "cricket", "ipl", "football", "fifa", "olympics",
        "match", "tournament", "series", "league",
        "player", "coach", "goal", "score",
        # Entertainment
        "movie", "film", "cinema", "actor", "actress",
        "box office", "trailer", "ott", "web series",
        # Lifestyle & health
        "health", "fitness", "diet", "lifestyle", "yoga",
        "hospital", "doctor", "disease",
        # City & local info
        "traffic", "road", "metro", "railway", "train",
        "airport", "flight", "school", "college",
        # Technology & gadgets
        "technology", "mobile", "smartphone", "launch",
        "review", "features",
        # General
        "tourism", "travel",
    ]

    # ── Political keywords (Gate 2) ──────────────────────────
    POLITICAL_KEYWORDS: List[str] = [
        # General politics
        "politics", "political", "governance", "administration",
        "democracy", "constitution", "constitutional", "sovereignty",
        # Government & institutions
        "government", "govt",
        "central government", "state government", "union government",
        "parliament", "lok sabha", "rajya sabha",
        "vidhan sabha", "vidhan parishad",
        "supreme court", "high court", "district court",
        "president", "vice president", "prime minister",
        "chief minister", "governor",
        "bureaucracy", "ias", "ips", "irs",
        "election commission", "eci", "niti aayog",
        "cbi", "ed", "income tax department",
        # Elections
        "election", "elections", "poll", "polls", "vote", "voting",
        "evm", "ballot", "campaign", "manifesto",
        "model code of conduct", "by-election",
        # Political parties
        "bjp", "bharatiya janata party",
        "congress", "indian national congress",
        "aap", "aam aadmi party",
        "cpi", "cpm", "left front",
        "tmc", "trinamool congress",
        "samajwadi party", "bahujan samaj party",
        "dmk", "aiadmk",
        "shiv sena", "ncp", "rjd", "jdu",
        "ysrcp", "trs", "bjd", "sad",
        # Political leaders
        "modi", "narendra modi",
        "rahul gandhi", "sonia gandhi", "priyanka gandhi",
        "amit shah", "yogi adityanath",
        "arvind kejriwal", "mamata banerjee",
        "nitish kumar", "uddhav thackeray",
        "mk stalin", "kcr", "naveen patnaik",
        # Laws, policy & economy
        "law", "bill", "act", "ordinance", "policy", "reform",
        "budget", "tax", "gst", "income tax",
        "privatization", "disinvestment",
        "inflation", "unemployment", "economy",
        # Social & ideological issues
        "reservation", "quota",
        "caste census", "scheduled caste", "scheduled tribe", "obc",
        "minority", "majority",
        "secularism", "communal",
        "hindutva", "nationalism", "anti-national",
        # Protests & movements
        "protest", "protests", "agitation", "strike",
        "farmer protest", "farm laws",
        "andolan", "demonstration",
        # Security & foreign policy
        "defence", "military", "armed forces",
        "border dispute", "kashmir", "article 370",
        "citizenship", "caa", "nrc",
        "foreign policy", "diplomacy",
    ]

    def __init__(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._non_political_patterns = self._compile(self.NON_POLITICAL_KEYWORDS)
        self._political_patterns = self._compile(self.POLITICAL_KEYWORDS)

    # ── Public API ───────────────────────────────────────────

    def classify(self, headline: str) -> FilterResult:
        """
        Classify a headline through the two-gate filter.

        Returns:
            FilterResult.NON_POLITICAL      – weather, sports, etc.
            FilterResult.NEUTRAL_POLITICAL   – political but no ideological framing
            FilterResult.BIASED_POLITICAL    – should be sent to BERT
        """
        text = headline.lower()

        if self._matches(text, self._non_political_patterns):
            return FilterResult.NON_POLITICAL

        if not self._matches(text, self._political_patterns):
            return FilterResult.NEUTRAL_POLITICAL

        return FilterResult.BIASED_POLITICAL

    def is_non_political(self, text: str) -> bool:
        """Legacy compatibility: returns True if headline is non-political."""
        return self.classify(text) == FilterResult.NON_POLITICAL

    def is_political(self, text: str) -> bool:
        """Legacy compatibility: returns True if headline contains political keywords."""
        return self._matches(text.lower(), self._political_patterns)

    # ── Internals ────────────────────────────────────────────

    @staticmethod
    def _compile(keywords: List[str]) -> list:
        """Pre-compile word-boundary regex patterns."""
        return [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]

    @staticmethod
    def _matches(text: str, patterns: list) -> bool:
        """Return True if any compiled pattern matches."""
        return any(p.search(text) for p in patterns)
