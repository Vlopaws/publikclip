"""Which language a piece of copy is in, to the only precision that matters.

The pipeline generates a title and a caption from a clip's own transcript,
and the prompt tells the model to answer in the transcript's language. A
French clip went out with an English caption anyway. Asking politely is not
a guarantee, and the cost of the failure lands on the audience: copy in the
wrong language reads as machine output to the exact people it is aimed at.

This is deliberately not a language identifier. It answers one question --
"are these two pieces of text in the same language" -- for the two
languages this operation actually publishes in, using function words, which
are the words a generated sentence cannot avoid and a topic cannot change.
A proper identifier would be a dependency and a model download to decide
something a hundred stopwords already decide.
"""

from __future__ import annotations

import re

# Function words, not vocabulary. "streamer", "clip" and "live" appear in
# both languages and in neither list.
_FRENCH = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "sont",
    "qui", "que", "quoi", "pas", "ne", "pour", "dans", "sur", "avec", "sans",
    "il", "elle", "ils", "elles", "on", "je", "tu", "nous", "vous", "ce",
    "cette", "ces", "son", "sa", "ses", "leur", "au", "aux", "en", "y",
    "plus", "moins", "tout", "tous", "toute", "mais", "donc", "ou", "où",
    "quand", "comme", "fait", "faire", "être", "avoir", "se", "lui", "moi",
}
_ENGLISH = {
    "the", "a", "an", "and", "is", "are", "was", "were", "of", "to", "in",
    "on", "at", "for", "with", "without", "that", "this", "these", "those",
    "it", "he", "she", "they", "we", "you", "i", "his", "her", "their",
    "not", "but", "so", "or", "when", "where", "which", "who", "what",
    "from", "by", "as", "has", "have", "had", "does", "do", "did", "into",
    # Added after measuring: "The streamer thanks donors for 100EUR gifts,
    # then keeps playing" scored one hit on the first list and so was called
    # unreadable -- the guard stayed silent on exactly the caption it was
    # written to catch. None of these exist in French, so widening costs no
    # accuracy on the other side.
    "then", "than", "there", "here", "now", "just", "also", "still", "after",
    "before", "while", "over", "under", "out", "up", "down", "about", "more",
    "most", "all", "some", "any", "very", "can", "will", "would", "should",
    "could", "get", "gets", "got", "keep", "keeps", "make", "makes", "go",
    "goes", "going", "say", "says", "back", "off", "own", "how", "why",
    "because", "if", "each", "other", "another", "every", "both", "much",
    "many", "such", "same", "own", "been", "being", "was", "will",
}

# Both lists share nothing, so a tie means the text is too short or too
# unusual to call -- a title of five words often is. Below this many
# function-word hits, no claim is made at all, because a wrong claim here
# silently deletes good copy.
MIN_HITS = 3

# How much one language has to lead. A French sentence quoting an English
# phrase ("let's go", "one shot") should still read as French.
LEAD = 1.5


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zà-öø-ÿ']+", (text or "").lower())


def language_of(text: str) -> str | None:
    """"fr", "en", or None when the text does not say clearly enough."""
    words = _words(text)
    if not words:
        return None
    fr = sum(1 for w in words if w in _FRENCH)
    en = sum(1 for w in words if w in _ENGLISH)
    if fr + en < MIN_HITS:
        return None
    if fr >= en * LEAD and fr > en:
        return "fr"
    if en >= fr * LEAD and en > fr:
        return "en"
    return None


def mismatched(copy: str, source: str) -> bool:
    """True only when both are readable AND they disagree.

    Deliberately asymmetric with the guard it serves: silence means keep the
    copy. Dropping a caption is a real loss, so it happens only on evidence,
    never on the absence of it.
    """
    a, b = language_of(copy), language_of(source)
    return bool(a and b and a != b)
