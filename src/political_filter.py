import re
from typing import List


# =========================================================
# NON-POLITICAL TOPICS (FIRST GATE – OVERRIDES EVERYTHING)
# =========================================================

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

    # General info
     "tourism", "travel"
]


# =========================================================
# POLITICAL KEYWORDS (ONLY CHECKED IF NOT NON-POLITICAL)
# =========================================================

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

    # Elections & process
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
    "foreign policy", "diplomacy"
]


# =========================================================
# INTERNAL SAFE MATCHER (WORD-BOUNDARY AWARE)
# =========================================================

def _contains_keyword(text: str, keywords: List[str]) -> bool:
    """
    Returns True if any keyword appears as a full word or phrase.
    Prevents false matches like 'sc' in 'score'.
    """
    text = text.lower()
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text):
            return True
    return False


# =========================================================
# PUBLIC API (USED BY STREAMLIT / INFERENCE)
# =========================================================

def is_non_political(text: str) -> bool:
    return _contains_keyword(text, NON_POLITICAL_KEYWORDS)


def is_political(text: str) -> bool:
    return _contains_keyword(text, POLITICAL_KEYWORDS)

