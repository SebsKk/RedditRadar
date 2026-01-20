"""Clustering module for RedditRadar.

Provides lightweight topic clustering without LLM:
- TF-IDF keyword extraction with proper cosine similarity
- Fixed taxonomy mapping with multi-label support
- Intensity and proxy signal detection with soft saturation
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# TOPIC-BASED CLUSTER TAXONOMY (Stable themes, not discussion formats)
# =============================================================================
# These clusters describe WHAT the topic is about, not HOW people discuss it.
# Post format (question, how-to, rant, win) is tracked separately as post_type.

TOPIC_CLUSTERS = {
    "pricing_monetization": {
        "name": "Pricing & Monetization",
        "keywords": [
            "pricing", "price", "monetization", "revenue", "mrr", "arr",
            "subscription", "freemium", "premium", "tier", "pricing model",
            "ltv", "cac", "unit economics", "margins", "profit", "cost",
            "budget", "roi", "charge", "fee", "rate", "discount", "upsell",
            "payment", "billing", "invoice", "paywall", "free trial",
        ],
    },
    "lead_gen_sales": {
        "name": "Lead Gen & Sales",
        "keywords": [
            "leads", "sales", "outreach", "cold email", "prospecting",
            "pipeline", "crm", "demo", "closing", "objections", "deal",
            "enterprise", "b2b", "b2c", "customers", "clients", "churn",
            "conversion", "funnel", "qualified", "prospect", "pitch",
            "proposal", "contract", "negotiation", "commission", "quota",
        ],
    },
    "marketing_growth": {
        "name": "Marketing & Growth",
        "keywords": [
            "marketing", "growth", "traction", "distribution", "acquisition",
            "viral", "organic", "paid ads", "seo", "content marketing",
            "referral", "word of mouth", "launch", "product hunt", "campaign",
            "brand", "awareness", "reach", "impressions", "engagement",
            "social media", "influencer", "affiliate", "partnership",
        ],
    },
    "execution_productivity": {
        "name": "Execution & Productivity",
        "keywords": [
            "focus", "execution", "prioritize", "productivity", "time management",
            "discipline", "consistency", "routine", "habits", "procrastination",
            "burnout", "overwhelmed", "stressed", "balance", "efficiency",
            "workflow", "process", "systems", "automation", "delegate",
            "outsource", "hire", "team", "manage", "deadline", "ship",
        ],
    },
    "local_services": {
        "name": "Local Services",
        "keywords": [
            "local", "neighborhood", "community", "door to door", "flyer",
            "lawn", "cleaning", "handyman", "plumber", "electrician",
            "contractor", "repair", "maintenance", "house", "home",
            "yard", "garden", "pet", "dog walking", "babysitting",
            "tutoring", "delivery", "errand", "moving", "junk removal",
        ],
    },
    "digital_services": {
        "name": "Digital Services & Freelance",
        "keywords": [
            "freelance", "agency", "client", "project", "scope", "creep",
            "retainer", "hourly", "fixed price", "web design", "development",
            "copywriting", "design", "logo", "branding", "consulting",
            "coaching", "virtual assistant", "bookkeeping", "social media manager",
            "video editing", "graphic design", "seo services", "ppc",
        ],
    },
    "digitization_nostalgia": {
        "name": "Digitization & Nostalgia",
        "keywords": [
            "digitize", "digitization", "vhs", "tape", "cassette", "vinyl",
            "film", "photos", "slides", "negatives", "old", "vintage",
            "retro", "nostalgia", "memories", "archive", "preserve",
            "convert", "transfer", "scan", "restore", "legacy", "family",
            "heirloom", "collectible", "antique",
        ],
    },
    "ecommerce_products": {
        "name": "E-commerce & Products",
        "keywords": [
            "ecommerce", "shopify", "amazon", "fba", "dropshipping",
            "product", "inventory", "supplier", "wholesale", "retail",
            "shipping", "fulfillment", "returns", "reviews", "listing",
            "marketplace", "etsy", "ebay", "physical product", "merch",
            "print on demand", "private label", "sourcing", "manufacturing",
        ],
    },
    "saas_software": {
        "name": "SaaS & Software",
        "keywords": [
            "saas", "software", "app", "platform", "tool", "api",
            "integration", "feature", "mvp", "beta", "users", "onboarding",
            "churn", "retention", "usage", "dashboard", "analytics",
            "subscription", "cloud", "hosting", "infrastructure", "tech stack",
            "no-code", "low-code", "ai", "automation", "bot",
        ],
    },
    "content_media": {
        "name": "Content & Media",
        "keywords": [
            "content", "blog", "youtube", "podcast", "newsletter", "course",
            "ebook", "info product", "creator", "audience", "subscribers",
            "followers", "views", "monetize", "sponsorship", "patreon",
            "membership", "community", "niche", "authority", "personal brand",
        ],
    },
    "finance_investing": {
        "name": "Finance & Investing",
        "keywords": [
            "invest", "investment", "stock", "crypto", "real estate",
            "passive income", "dividend", "portfolio", "trading", "market",
            "fund", "angel", "vc", "fundraising", "valuation", "equity",
            "debt", "loan", "credit", "savings", "retirement", "tax",
        ],
    },
    "legal_compliance": {
        "name": "Legal & Compliance",
        "keywords": [
            "legal", "law", "lawyer", "attorney", "contract", "agreement",
            "liability", "insurance", "llc", "corporation", "trademark",
            "copyright", "patent", "ip", "terms", "privacy", "gdpr",
            "compliance", "regulation", "license", "permit", "lawsuit",
        ],
    },
    "general": {
        "name": "General",
        "keywords": [],
    },
}

# =============================================================================
# POST TYPE DETECTION (How people discuss - separate from topic)
# =============================================================================
# Post type describes the FORMAT of discussion, tracked as a separate field

POST_TYPES = {
    "question": {
        "name": "Question",
        "keywords": [
            "how", "what", "why", "when", "where", "who", "which",
            "anyone", "help", "advice", "recommend", "suggest", "?",
            "confused", "stuck", "struggling", "need", "looking for",
        ],
    },
    "howto": {
        "name": "How-To / Guide",
        "keywords": [
            "guide", "tutorial", "step by step", "how to", "walkthrough",
            "instructions", "tips", "here's how", "method", "process",
        ],
    },
    "win": {
        "name": "Win / Success Story",
        "keywords": [
            "finally", "achieved", "success", "milestone", "hit", "reached",
            "excited", "proud", "made it", "first sale", "profitable",
            "quit my job", "revenue", "customers", "launched",
        ],
    },
    "rant": {
        "name": "Rant / Frustration",
        "keywords": [
            "rant", "frustrated", "annoyed", "sick of", "tired of",
            "hate", "terrible", "awful", "worst", "ridiculous", "vent",
            "complain", "unfair", "scam", "warning",
        ],
    },
    "discussion": {
        "name": "Discussion / Opinion",
        "keywords": [
            "thoughts", "opinion", "think", "debate", "controversial",
            "unpopular", "change my mind", "perspective", "view",
        ],
    },
    "news": {
        "name": "News / Update",
        "keywords": [
            "news", "announcement", "update", "launched", "released",
            "breaking", "just", "today", "new", "introduced",
        ],
    },
    "resource": {
        "name": "Resource / Tool",
        "keywords": [
            "resource", "tool", "list", "collection", "found this",
            "sharing", "free", "useful", "check out", "recommend",
        ],
    },
}

# Alias for backwards compatibility
CORE_CLUSTERS = TOPIC_CLUSTERS

# =============================================================================
# PRECOMPUTED CLUSTER DATA (for performance)
# =============================================================================

def _build_cluster_token_sets() -> dict[str, set[str]]:
    """Precompute tokenized keyword sets for each cluster."""
    result = {}
    for cluster_id, cluster_data in CORE_CLUSTERS.items():
        tokens = set()
        for kw in cluster_data["keywords"]:
            # Tokenize each keyword phrase
            kw_tokens = kw.lower().split()
            tokens.update(kw_tokens)
        result[cluster_id] = tokens
    return result


def _build_cluster_tfidf_vectors() -> dict[str, dict[str, float]]:
    """Build TF-IDF vectors for each cluster based on their keywords.

    Each cluster is treated as a document containing its keywords.
    """
    # Count document frequency for each term
    all_terms: set[str] = set()
    cluster_term_counts: dict[str, Counter] = {}

    for cluster_id, cluster_data in CORE_CLUSTERS.items():
        if cluster_id == "general":
            continue
        term_counter = Counter()
        for kw in cluster_data["keywords"]:
            # Split multi-word keywords into tokens
            tokens = kw.lower().split()
            term_counter.update(tokens)
            all_terms.update(tokens)
        cluster_term_counts[cluster_id] = term_counter

    # Compute IDF for each term
    num_clusters = len(cluster_term_counts)
    doc_freq: dict[str, int] = Counter()
    for term in all_terms:
        for counter in cluster_term_counts.values():
            if counter[term] > 0:
                doc_freq[term] += 1

    idf: dict[str, float] = {}
    for term in all_terms:
        idf[term] = math.log(num_clusters / (1 + doc_freq[term])) + 1

    # Build TF-IDF vectors
    vectors: dict[str, dict[str, float]] = {}
    for cluster_id, term_counter in cluster_term_counts.items():
        total_terms = sum(term_counter.values())
        if total_terms == 0:
            vectors[cluster_id] = {}
            continue

        vec = {}
        for term, count in term_counter.items():
            tf = count / total_terms
            vec[term] = tf * idf.get(term, 1.0)

        # Normalize vector
        magnitude = math.sqrt(sum(v * v for v in vec.values()))
        if magnitude > 0:
            vec = {t: v / magnitude for t, v in vec.items()}

        vectors[cluster_id] = vec

    return vectors


# Precompute on module load
CLUSTER_TOKEN_SETS = _build_cluster_token_sets()
CLUSTER_TFIDF_VECTORS = _build_cluster_tfidf_vectors()


# =============================================================================
# PAIN/INTENSITY LANGUAGE
# =============================================================================

PAIN_TOKENS = [
    "desperate", "stuck", "burned out", "burnout", "overwhelmed", "frustrated",
    "losing money", "losing time", "wasting", "struggling", "painful", "nightmare",
    "hate", "terrible", "awful", "broken", "failed", "failing", "mistake",
    "regret", "anxiety", "stressed", "exhausted", "confused", "lost",
]

# =============================================================================
# TEMPLATE-SPECIFIC PROXY LEXICONS (measure engagement/impact signals)
# =============================================================================

# High engagement signals (applies to all templates)
ENGAGEMENT_LEXICON = [
    "important", "urgent", "critical", "must", "need", "essential",
    "significant", "major", "serious", "crucial", "vital", "key",
    "problem", "issue", "challenge", "struggle", "difficult", "hard",
    "amazing", "incredible", "awesome", "fantastic", "great", "best",
    "worst", "terrible", "awful", "disaster", "crisis", "emergency",
]

# Content-focused signals (content_ideas template)
WTP_LEXICON_CONTENT = [
    "viral", "popular", "trending", "engagement", "views", "clicks",
    "shares", "likes", "followers", "audience", "reach", "growth",
    "content", "post", "article", "video", "podcast", "blog",
    "strategy", "marketing", "brand", "influence", "creator",
]

# Career-focused signals (career_intel template)
WTP_LEXICON_CAREER = [
    "interview", "hiring", "salary", "compensation", "offer", "portfolio",
    "skills", "learn", "certif", "bootcamp", "job", "position", "role",
    "promotion", "manager", "senior", "junior", "intern",
    "experience", "resume", "cv", "recruiter", "employer",
]

# Research-focused signals (deep_research template)
WTP_LEXICON_RESEARCH = [
    "evidence", "study", "data", "source", "research", "paper", "journal",
    "uncertain", "hypothesis", "experiment", "methodology", "peer review",
    "citation", "findings", "conclusion", "analysis", "scientific",
    "verified", "confirmed", "debunked", "factual", "accurate",
]

# Trend-focused signals (trend_radar template)
WTP_LEXICON_TREND = [
    "emerging", "new", "rising", "growing", "trend", "future", "next",
    "adoption", "mainstream", "early", "cutting edge", "innovative",
    "disrupt", "shift", "change", "transform", "prediction", "forecast",
]

# Industry/business signals (industry_intel template)
WTP_LEXICON_INDUSTRY = [
    "market", "industry", "competitor", "player", "stakeholder", "buyer",
    "seller", "vendor", "regulation", "compliance", "policy", "merger",
    "acquisition", "funding", "ipo", "valuation", "revenue", "profit",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PostFeatures:
    """Computed features for a single post."""
    post_id: str
    cluster: str  # Primary topic cluster ID
    cluster_name: str  # Primary topic cluster name
    post_type: str = "discussion"  # Post format: question, howto, win, rant, etc.
    post_type_name: str = "Discussion"  # Human-readable post type
    secondary_clusters: list[str] = field(default_factory=list)  # Multi-label support
    keywords: list[str] = field(default_factory=list)
    keyword_overlap: float = 0.0  # 0-1 score based on goal overlap
    intensity_score: float = 0.0  # 0-1 based on pain language
    proxy_score: float = 0.0  # 0-1 based on template-specific lexicon
    base_score: float = 0.0  # Combined score
    cluster_similarity: float = 0.0  # Cosine similarity to best cluster
    urls: list[str] = field(default_factory=list)
    permalink: str = ""  # Reddit post URL


@dataclass
class ClusterResult:
    """Aggregated metrics for a cluster."""
    cluster_id: str
    cluster_name: str
    frequency_posts: int
    frequency_comments: int
    engagement_score_sum: int
    engagement_comments_sum: int
    avg_age_hours: float  # Average age of posts in hours
    intensity_score: float  # avg across posts
    proxy_score: float  # avg across posts
    signal_strength: int = 3  # 1-5 stars based on freq + subreddits
    unique_subreddits: int = 1  # Number of unique subreddits
    top_urls: list[str] = field(default_factory=list)  # Reddit permalinks
    top_post_ids: list[str] = field(default_factory=list)
    post_type_breakdown: dict = field(default_factory=dict)  # e.g., {"question": 3, "win": 2}
    key_signals: list[str] = field(default_factory=list)  # Top cluster-level signals (bigrams/terms)
    value_score: float = 0.0  # LLM-assigned value score (0-10) for heatmap

    # For backwards compatibility
    @property
    def recency_avg_hours(self) -> float:
        return self.avg_age_hours


# =============================================================================
# KEYWORD EXTRACTION (TF-IDF)
# =============================================================================

# Common stopwords to filter from keywords
STOPWORDS = {
    # Articles, pronouns, prepositions
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
    'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
    'those', 'what', 'which', 'who', 'whom', 'any', 'both', 'each',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its', 'they',
    'them', 'their', 'he', 'she', 'him', 'her', 'his', 'hers',
    'about', 'also', 'back', 'been', 'being', 'come', 'even', 'get',
    'go', 'going', 'got', 'know', 'like', 'make', 'much', 'now', 'out',
    'over', 'really', 'say', 'see', 'take', 'think', 'time', 'want',
    'way', 'well', 'work', 'year', 'new', 'good', 'first', 'last',
    'long', 'great', 'little', 'own', 'old', 'right', 'big', 'high',
    'different', 'small', 'large', 'next', 'early', 'young', 'important',
    'few', 'public', 'bad', 'same', 'able', 'dont', 'ive', 'im', 'youre',
    'one', 'two', 'three', 'many', 'still', 'something', 'anything',
    'everything', 'nothing', 'someone', 'anyone', 'everyone', 'people',
    'thing', 'things', 'lot', 'kind', 'point', 'part', 'place', 'case',
    'week', 'company', 'system', 'program', 'question', 'hand', 'number',
    'https', 'http', 'www', 'com', 'reddit', 'subreddit', 'post', 'comment',
    # Utility tokens and linguistic artifacts
    'end', 'note', 'similar', 'format', 'example', 'basically', 'actually',
    'probably', 'maybe', 'perhaps', 'definitely', 'certainly', 'usually',
    'always', 'never', 'sometimes', 'often', 'ago', 'months', 'days', 'years',
    'hours', 'minutes', 'seconds', 'today', 'yesterday', 'tomorrow',
    'looking', 'thinking', 'wondering', 'trying', 'getting', 'making',
    'starting', 'using', 'seeing', 'finding', 'going', 'coming', 'being',
    'having', 'doing', 'saying', 'asking', 'telling', 'showing', 'keeping',
    'edit', 'update', 'thanks', 'thank', 'please', 'sorry', 'hope', 'guess',
    'sure', 'pretty', 'quite', 'rather', 'really', 'truly', 'simply',
    'imo', 'imho', 'tbh', 'fwiw', 'btw', 'afaik', 'iirc', 'etc', 'ie', 'eg',
}

# Location/country names to downweight in signals (noisy in global discussions)
LOCATION_BLACKLIST = {
    # Countries
    'usa', 'uk', 'us', 'canada', 'australia', 'germany', 'france', 'italy',
    'spain', 'japan', 'china', 'india', 'brazil', 'mexico', 'russia',
    'netherlands', 'belgium', 'switzerland', 'sweden', 'norway', 'denmark',
    'finland', 'poland', 'austria', 'ireland', 'portugal', 'greece',
    'argentina', 'chile', 'colombia', 'peru', 'egypt', 'turkey', 'israel',
    'saudi', 'uae', 'singapore', 'malaysia', 'indonesia', 'philippines',
    'thailand', 'vietnam', 'korea', 'taiwan', 'hong', 'kong', 'zealand',
    # Demonyms
    'american', 'british', 'canadian', 'australian', 'german', 'french',
    'italian', 'spanish', 'japanese', 'chinese', 'indian', 'brazilian',
    'mexican', 'russian', 'dutch', 'swiss', 'swedish', 'european', 'asian',
    # Cities (common)
    'london', 'paris', 'berlin', 'tokyo', 'beijing', 'shanghai', 'mumbai',
    'delhi', 'sydney', 'melbourne', 'toronto', 'vancouver', 'montreal',
    'chicago', 'boston', 'seattle', 'austin', 'denver', 'miami', 'atlanta',
    # Generic location words
    'city', 'state', 'country', 'region', 'area', 'local', 'national',
    'international', 'global', 'worldwide', 'domestic', 'foreign',
    'moved', 'moving', 'living', 'leaving', 'staying', 'visiting',
}


def tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric."""
    if not text:
        return []
    text = text.lower()
    # Keep alphanumeric and some special chars
    tokens = re.findall(r'\b[a-z][a-z0-9]*(?:[-/][a-z0-9]+)*\b', text)
    return tokens


# Common verbs that make bigrams less meaningful
STOP_VERBS = {
    'give', 'gave', 'giving', 'get', 'got', 'getting', 'make', 'made', 'making',
    'do', 'did', 'doing', 'done', 'take', 'took', 'taking', 'put', 'putting',
    'use', 'used', 'using', 'find', 'found', 'try', 'tried', 'trying',
    'look', 'looked', 'looking', 'need', 'needed', 'want', 'wanted',
    'tell', 'told', 'ask', 'asked', 'let', 'help', 'helped', 'keep', 'kept',
    'start', 'started', 'stop', 'stopped', 'run', 'running', 'feel', 'felt',
    'seem', 'seemed', 'become', 'became', 'show', 'showed', 'leave', 'left',
    'call', 'called', 'read', 'write', 'written', 'bring', 'brought',
    'choice', 'choose', 'chose', 'choosing', 'definitive', 'review',
}


def extract_bigrams(tokens: list[str]) -> list[str]:
    """Extract bigrams from a list of tokens, filtering stopwords and stop-verbs."""
    if len(tokens) < 2:
        return []

    bigrams = []
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        # Skip if either token is a stopword or too short
        if t1 in STOPWORDS or t2 in STOPWORDS:
            continue
        if len(t1) < 3 or len(t2) < 3:
            continue
        # Skip location words in bigrams
        if t1 in LOCATION_BLACKLIST or t2 in LOCATION_BLACKLIST:
            continue
        # Skip common verbs that make bigrams less meaningful
        if t1 in STOP_VERBS or t2 in STOP_VERBS:
            continue
        bigrams.append(f"{t1} {t2}")
    return bigrams


def extract_cluster_signals(
    texts_with_engagement: list[tuple[str, int]],
    top_n: int = 5,
    min_doc_frequency: int = 2,
) -> list[str]:
    """Extract top signals for a cluster using tf * engagement weighting.

    Prefers bigrams over unigrams for more meaningful signals.
    Filters out location names and generic words.
    Requires terms to appear in multiple documents (posts) to avoid noise.

    Args:
        texts_with_engagement: List of (text, engagement_score) tuples
        top_n: Number of top signals to return
        min_doc_frequency: Minimum number of documents a term must appear in

    Returns:
        List of top signal terms/bigrams
    """
    if not texts_with_engagement:
        return []

    num_docs = len(texts_with_engagement)

    # Track both scores and document frequency
    bigram_scores: dict[str, float] = {}
    bigram_doc_freq: dict[str, int] = {}
    unigram_scores: dict[str, float] = {}
    unigram_doc_freq: dict[str, int] = {}

    total_engagement = sum(max(e, 1) for _, e in texts_with_engagement)

    for text, engagement in texts_with_engagement:
        tokens = tokenize(text)
        if not tokens:
            continue

        # Normalize engagement to a 0-1 weight
        eng_weight = max(engagement, 1) / total_engagement if total_engagement > 0 else 1

        # Count term frequencies in this document
        token_counts = Counter(tokens)
        total = len(tokens)

        # Track unique terms in this document (for doc frequency)
        seen_unigrams: set[str] = set()
        seen_bigrams: set[str] = set()

        # Score unigrams
        for token, count in token_counts.items():
            if len(token) < 3 or token in STOPWORDS:
                continue
            if token in LOCATION_BLACKLIST:
                continue
            # TF * engagement weight
            tf = count / total
            score = tf * (1 + eng_weight * 2)  # Boost by engagement
            unigram_scores[token] = unigram_scores.get(token, 0) + score
            if token not in seen_unigrams:
                unigram_doc_freq[token] = unigram_doc_freq.get(token, 0) + 1
                seen_unigrams.add(token)

        # Score bigrams
        bigrams = extract_bigrams(tokens)
        bigram_counts = Counter(bigrams)
        for bigram, count in bigram_counts.items():
            tf = count / max(len(bigrams), 1)
            score = tf * (1 + eng_weight * 2)
            bigram_scores[bigram] = bigram_scores.get(bigram, 0) + score
            if bigram not in seen_bigrams:
                bigram_doc_freq[bigram] = bigram_doc_freq.get(bigram, 0) + 1
                seen_bigrams.add(bigram)

    # Adjust min_doc_frequency if we have few documents
    effective_min_df = min(min_doc_frequency, max(1, num_docs // 3))

    # Filter to terms appearing in enough documents
    valid_bigrams = {
        bg: score for bg, score in bigram_scores.items()
        if bigram_doc_freq.get(bg, 0) >= effective_min_df
    }
    valid_unigrams = {
        tok: score for tok, score in unigram_scores.items()
        if unigram_doc_freq.get(tok, 0) >= effective_min_df
    }

    # Combine results, preferring bigrams (more meaningful)
    signals = []

    # Add top bigrams first (up to 3)
    sorted_bigrams = sorted(valid_bigrams.items(), key=lambda x: x[1], reverse=True)
    for bigram, score in sorted_bigrams[:3]:
        if score > 0.01:  # Minimum threshold
            signals.append(bigram)

    # Fill remaining with top unigrams
    sorted_unigrams = sorted(valid_unigrams.items(), key=lambda x: x[1], reverse=True)
    for token, score in sorted_unigrams:
        if len(signals) >= top_n:
            break
        # Skip if already covered by a bigram
        if any(token in bg for bg in signals):
            continue
        if score > 0.01:
            signals.append(token)

    return signals[:top_n]


def extract_keywords(
    text: str,
    goal_tokens: set[str],
    top_n: int = 10,
) -> tuple[list[str], float]:
    """Extract keywords from text and compute overlap with goal.

    Args:
        text: Input text
        goal_tokens: Set of unique tokens from goal prompt
        top_n: Number of top keywords to extract

    Returns:
        Tuple of (keywords, overlap_score)
    """
    tokens = tokenize(text)
    if not tokens:
        return [], 0.0

    # Count token frequencies
    token_counts = Counter(tokens)

    # Build TF-IDF vector for this text
    total_tokens = len(tokens)
    scored_tokens = []

    for token, count in token_counts.items():
        # Skip short tokens and stopwords
        if len(token) < 3 or token in STOPWORDS:
            continue

        # TF: term frequency
        tf = count / total_tokens

        # Boost factor based on relevance signals
        boost = 1.0

        # Boost if in goal tokens (exact match)
        if token in goal_tokens:
            boost *= 3.0

        # Boost if matches any cluster keyword token (exact match)
        for cluster_id, token_set in CLUSTER_TOKEN_SETS.items():
            if cluster_id == "general":
                continue
            if token in token_set:
                boost *= 2.0
                break

        score = tf * boost
        scored_tokens.append((token, score))

    # Sort by score and take top N
    scored_tokens.sort(key=lambda x: x[1], reverse=True)
    keywords = [t[0] for t in scored_tokens[:top_n]]

    # Compute overlap: intersection with goal divided by min(top_n, goal size)
    # This prevents near-zero overlap from huge goal prompts
    if goal_tokens and keywords:
        keywords_set = set(keywords)
        hits = len(keywords_set & goal_tokens)
        # Normalize by the smaller of top_n or unique goal tokens (capped at 20)
        divisor = min(top_n, len(goal_tokens), 20)
        overlap = hits / divisor if divisor > 0 else 0.0
    else:
        overlap = 0.0

    return keywords, min(1.0, overlap)


# =============================================================================
# CLUSTER ASSIGNMENT (TF-IDF Cosine Similarity)
# =============================================================================

def compute_text_tfidf(tokens: list[str]) -> dict[str, float]:
    """Compute TF-IDF vector for a list of tokens."""
    if not tokens:
        return {}

    token_counts = Counter(tokens)
    total = len(tokens)

    vec = {}
    for token, count in token_counts.items():
        tf = count / total
        # Use simple IDF approximation (boost rare-in-clusters terms)
        # Check how many clusters contain this token
        cluster_count = sum(1 for ts in CLUSTER_TOKEN_SETS.values() if token in ts)
        idf = math.log(10 / (1 + cluster_count)) + 1  # 10 clusters
        vec[token] = tf * idf

    # Normalize
    magnitude = math.sqrt(sum(v * v for v in vec.values()))
    if magnitude > 0:
        vec = {t: v / magnitude for t, v in vec.items()}

    return vec


def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec1 or not vec2:
        return 0.0

    # Dot product (only shared keys matter since vectors are normalized)
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in vec1.keys() & vec2.keys())
    return dot


def assign_cluster(
    text: str,
    keywords: list[str],
    min_similarity: float = 0.05,
    secondary_threshold: float = 0.7,
) -> tuple[str, str, list[str], float]:
    """Assign text to the best-matching cluster using TF-IDF cosine similarity.

    Args:
        text: Full text to analyze
        keywords: Extracted keywords
        min_similarity: Minimum similarity to assign (otherwise general)
        secondary_threshold: If second-best cluster has similarity >= this * best, include it

    Returns:
        Tuple of (cluster_id, cluster_name, secondary_cluster_ids, best_similarity)
    """
    tokens = tokenize(text) if text else []
    text_vec = compute_text_tfidf(tokens)

    if not text_vec:
        return "general", CORE_CLUSTERS["general"]["name"], [], 0.0

    # Compute similarity to each cluster
    similarities = []
    for cluster_id, cluster_vec in CLUSTER_TFIDF_VECTORS.items():
        sim = cosine_similarity(text_vec, cluster_vec)
        similarities.append((cluster_id, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    best_cluster_id, best_sim = similarities[0]

    # Check if similarity is too low
    if best_sim < min_similarity:
        return "general", CORE_CLUSTERS["general"]["name"], [], 0.0

    # Find secondary clusters (multi-label)
    secondary = []
    if len(similarities) > 1:
        for cluster_id, sim in similarities[1:3]:  # Top 2 secondaries
            if sim >= best_sim * secondary_threshold:
                secondary.append(cluster_id)

    return best_cluster_id, CORE_CLUSTERS[best_cluster_id]["name"], secondary, best_sim


def detect_post_type(title: str, body: str = "") -> tuple[str, str]:
    """Detect the type/format of a post (question, how-to, win, rant, etc.).

    Args:
        title: Post title (primary signal)
        body: Post body (secondary signal)

    Returns:
        Tuple of (post_type_id, post_type_name)
    """
    text = f"{title} {body}".lower()

    # Score each post type
    scores = []
    for type_id, type_data in POST_TYPES.items():
        score = 0
        for kw in type_data["keywords"]:
            if kw in text:
                # Title matches count more
                if kw in title.lower():
                    score += 2
                else:
                    score += 1
        scores.append((type_id, type_data["name"], score))

    # Sort by score
    scores.sort(key=lambda x: x[2], reverse=True)

    # Return best match if score > 0, otherwise default to discussion
    if scores[0][2] > 0:
        return scores[0][0], scores[0][1]
    return "discussion", "Discussion"


# =============================================================================
# INTENSITY & PROXY SCORING (with soft saturation)
# =============================================================================

def soft_saturate(count: int, k: float = 0.5) -> float:
    """Soft saturation using exponential decay: 1 - exp(-k * count).

    This gives better resolution than linear capping:
    - k=0.5: count=1 -> 0.39, count=3 -> 0.78, count=5 -> 0.92, count=8 -> 0.98
    """
    return 1.0 - math.exp(-k * count)


def compute_intensity(text: str) -> float:
    """Compute intensity score based on pain language (0-1).

    Uses phrase matching for multi-word pain tokens and soft saturation.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    pain_count = sum(1 for token in PAIN_TOKENS if token in text_lower)

    # Soft saturation: more gradual curve
    return soft_saturate(pain_count, k=0.4)


def compute_proxy_score(
    text: str,
    template_id: str,
) -> float:
    """Compute engagement/impact proxy score based on template (0-1).

    Uses phrase matching and soft saturation for better resolution.
    """
    if not text:
        return 0.0

    text_lower = text.lower()

    # Base engagement lexicon applies to all templates
    lexicon = list(ENGAGEMENT_LEXICON)

    # Add template-specific lexicon
    if template_id == "content_ideas":
        lexicon.extend(WTP_LEXICON_CONTENT)
    elif template_id == "industry_intel":
        lexicon.extend(WTP_LEXICON_INDUSTRY)
    elif template_id == "career_intel":
        lexicon.extend(WTP_LEXICON_CAREER)
    elif template_id == "deep_research":
        lexicon.extend(WTP_LEXICON_RESEARCH)
    elif template_id == "trend_radar":
        lexicon.extend(WTP_LEXICON_TREND)
    else:
        # Default: combine content and trend
        lexicon.extend(WTP_LEXICON_CONTENT)
        lexicon.extend(WTP_LEXICON_TREND)

    # Count matches (phrase matching for multi-word terms)
    match_count = sum(1 for token in lexicon if token.lower() in text_lower)

    # Soft saturation with k=0.3 (needs more matches to saturate)
    return soft_saturate(match_count, k=0.3)


# =============================================================================
# POST FEATURE COMPUTATION
# =============================================================================

def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    if not text:
        return []
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    # Dedupe while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls[:5]  # Max 5 URLs


def compute_post_features(
    post_id: str,
    title: str,
    body: str,
    comments_text: str,
    goal_prompt: str,
    template_id: str,
    permalink: str = "",
) -> PostFeatures:
    """Compute all features for a single post.

    Args:
        post_id: Unique post identifier
        title: Post title
        body: Post body text
        comments_text: Combined comment text
        goal_prompt: Analysis goal for relevance scoring
        template_id: Template being used
        permalink: Reddit permalink URL for this post

    Returns:
        PostFeatures with topic cluster and post type
    """
    # Combine text
    full_text = f"{title} {body} {comments_text}"

    # Tokenize goal for overlap computation (use set for exact matching)
    goal_tokens = set(tokenize(goal_prompt))

    # Extract keywords and compute overlap
    keywords, overlap = extract_keywords(full_text, goal_tokens)

    # Assign TOPIC cluster using TF-IDF cosine similarity
    cluster_id, cluster_name, secondary_clusters, cluster_sim = assign_cluster(
        full_text, keywords
    )

    # Detect POST TYPE (format: question, how-to, win, rant, etc.)
    post_type_id, post_type_name = detect_post_type(title, body)

    # Compute intensity
    intensity = compute_intensity(full_text)

    # Compute proxy score
    proxy = compute_proxy_score(full_text, template_id)

    # Compute base score with cluster similarity as a factor
    base_score = (
        overlap * 0.25 +
        intensity * 0.2 +
        proxy * 0.25 +
        cluster_sim * 0.2 +
        0.1  # Base
    )

    return PostFeatures(
        post_id=post_id,
        cluster=cluster_id,
        cluster_name=cluster_name,
        post_type=post_type_id,
        post_type_name=post_type_name,
        secondary_clusters=secondary_clusters,
        keywords=keywords,
        keyword_overlap=overlap,
        intensity_score=intensity,
        proxy_score=proxy,
        base_score=base_score,
        cluster_similarity=cluster_sim,
        urls=[],  # Deprecated - use permalink instead
        permalink=permalink,
    )


# =============================================================================
# CLUSTER AGGREGATION
# =============================================================================

def aggregate_cluster_metrics(
    post_features: list[PostFeatures],
    posts_data: list[dict],  # List of post dicts with score, num_comments, created_utc, url, subreddit, title, body
) -> list[ClusterResult]:
    """Aggregate post features into cluster metrics.

    Args:
        post_features: List of computed PostFeatures
        posts_data: List of dicts with keys: post_id, score, num_comments, created_utc, url, subreddit, title, body

    Returns:
        List of ClusterResult sorted by engagement
    """
    import time

    # Build lookup for post data
    posts_lookup = {p["post_id"]: p for p in posts_data}

    # Group by cluster (include secondary cluster assignments with lower weight)
    clusters: dict[str, list[tuple[PostFeatures, float]]] = {}

    for pf in post_features:
        # Primary cluster gets weight 1.0
        if pf.cluster not in clusters:
            clusters[pf.cluster] = []
        clusters[pf.cluster].append((pf, 1.0))

        # Secondary clusters get weight 0.5
        for sec_cluster in pf.secondary_clusters:
            if sec_cluster not in clusters:
                clusters[sec_cluster] = []
            clusters[sec_cluster].append((pf, 0.5))

    results = []
    current_time = time.time()

    # First pass: collect raw metrics for normalization
    raw_metrics = []

    for cluster_id, weighted_features in clusters.items():
        if cluster_id == "general" and len(weighted_features) == 0:
            continue

        # Aggregate metrics (weighted by assignment confidence)
        frequency_posts = sum(w for _, w in weighted_features)
        engagement_score_sum = 0
        engagement_comments_sum = 0
        age_sum = 0
        intensity_sum = 0
        proxy_sum = 0
        post_urls = []  # Use Reddit permalinks
        post_ids = []
        post_type_counts: dict[str, int] = {}
        unique_subreddits: set[str] = set()
        texts_for_signals: list[tuple[str, int]] = []  # (text, engagement) for key_signals
        weight_sum = 0

        for pf, weight in weighted_features:
            post_data = posts_lookup.get(pf.post_id, {})
            post_engagement = post_data.get("score", 0) + post_data.get("num_comments", 0)
            engagement_score_sum += int(post_data.get("score", 0) * weight)
            engagement_comments_sum += int(post_data.get("num_comments", 0) * weight)

            created = post_data.get("created_utc", current_time)
            age_hours = (current_time - created) / 3600
            age_sum += age_hours * weight

            intensity_sum += pf.intensity_score * weight
            proxy_sum += pf.proxy_score * weight

            if weight >= 1.0:  # Only primary assignments contribute
                # Use Reddit permalink
                post_url = post_data.get("url", "") or pf.permalink
                if post_url:
                    post_urls.append((post_url, post_data.get("score", 0)))
                post_ids.append(pf.post_id)

                # Track post type breakdown
                post_type_counts[pf.post_type] = post_type_counts.get(pf.post_type, 0) + 1

                # Track unique subreddits
                subreddit = post_data.get("subreddit", "")
                if subreddit:
                    unique_subreddits.add(subreddit.lower())

                # Collect text for key signals (title + truncated body)
                title = post_data.get("title", "")
                body = post_data.get("body_truncated", post_data.get("body", ""))[:500]
                text = f"{title} {body}".strip()
                if text:
                    texts_for_signals.append((text, post_engagement))

            weight_sum += weight

        # Sort post URLs by engagement score and take top 3
        post_urls.sort(key=lambda x: x[1], reverse=True)
        top_urls = [url for url, _ in post_urls[:3]]

        # Extract cluster-level key signals
        key_signals = extract_cluster_signals(texts_for_signals, top_n=5)

        raw_metrics.append({
            "cluster_id": cluster_id,
            "frequency_posts": int(round(frequency_posts)),
            "engagement_score_sum": engagement_score_sum,
            "engagement_comments_sum": engagement_comments_sum,
            "avg_age_hours": age_sum / weight_sum if weight_sum > 0 else 0,
            "intensity_score": intensity_sum / weight_sum if weight_sum > 0 else 0,
            "proxy_score": proxy_sum / weight_sum if weight_sum > 0 else 0,
            "top_urls": top_urls,
            "top_post_ids": post_ids[:5],
            "post_type_breakdown": post_type_counts,
            "unique_subreddits": len(unique_subreddits),
            "key_signals": key_signals,
        })

    # Second pass: compute signal_strength based on freq + subreddits
    # New formula:
    # 5★: freq >= 8 AND subreddits >= 2
    # 4★: freq >= 5
    # 3★: freq 3-4
    # 2★: freq == 2
    # 1★: freq == 1
    for m in raw_metrics:
        freq = m["frequency_posts"]
        subs = m["unique_subreddits"]

        if freq >= 8 and subs >= 2:
            signal = 5
        elif freq >= 5:
            signal = 4
        elif freq >= 3:
            signal = 3
        elif freq >= 2:
            signal = 2
        else:
            signal = 1

        results.append(ClusterResult(
            cluster_id=m["cluster_id"],
            cluster_name=CORE_CLUSTERS.get(m["cluster_id"], {"name": m["cluster_id"]})["name"],
            frequency_posts=m["frequency_posts"],
            frequency_comments=0,
            engagement_score_sum=m["engagement_score_sum"],
            engagement_comments_sum=m["engagement_comments_sum"],
            avg_age_hours=m["avg_age_hours"],
            intensity_score=m["intensity_score"],
            proxy_score=m["proxy_score"],
            signal_strength=signal,
            unique_subreddits=m["unique_subreddits"],
            top_urls=m["top_urls"],
            top_post_ids=m["top_post_ids"],
            post_type_breakdown=m["post_type_breakdown"],
            key_signals=m["key_signals"],
        ))

    # Sort by engagement
    results.sort(key=lambda x: x.engagement_score_sum + x.engagement_comments_sum, reverse=True)

    return results


def intensity_label(score: float) -> str:
    """Convert intensity score to label."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"


def _format_stars(n: int) -> str:
    """Format signal strength as stars."""
    return "★" * n + "☆" * (5 - n)


def format_cluster_summary(clusters: list[ClusterResult]) -> str:
    """Format cluster results as markdown."""
    lines = ["## Cluster Summary\n"]

    for cluster in clusters:
        if cluster.frequency_posts == 0:
            continue

        # Header with signal strength stars
        stars = _format_stars(cluster.signal_strength)
        lines.append(f"### {cluster.cluster_name} {stars}")

        # Frequency and subreddit spread
        sub_info = f" across {cluster.unique_subreddits} subreddit(s)" if cluster.unique_subreddits > 1 else ""
        lines.append(f"- **Frequency:** {cluster.frequency_posts} posts{sub_info}")

        # Format engagement
        score_k = cluster.engagement_score_sum / 1000
        if score_k >= 1:
            lines.append(f"- **Engagement:** {score_k:.1f}k score, {cluster.engagement_comments_sum} comments")
        else:
            lines.append(f"- **Engagement:** {cluster.engagement_score_sum} score, {cluster.engagement_comments_sum} comments")

        lines.append(f"- **Avg age:** {cluster.avg_age_hours:.0f}h")

        # Key signals (cluster-level bigrams/terms)
        if cluster.key_signals:
            signals_str = ", ".join(cluster.key_signals[:5])
            lines.append(f"- **Key signals:** {signals_str}")

        # WTP/Impact proxy
        lines.append(f"- **WTP/Impact proxy:** {intensity_label(cluster.proxy_score)}")

        if cluster.top_urls:
            lines.append(f"- **Top sources:** " + ", ".join(f"[link]({url})" for url in cluster.top_urls[:3]))

        lines.append("")

    return "\n".join(lines)
