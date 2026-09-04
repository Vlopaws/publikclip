"""What an unattended run must not publish by itself.

The operator asked for autonomous public posting: pick the clips, post them,
do not ask. That is a reasonable thing to want and it removes the person who
was, until now, the last check before something went out.

It also removes them from a specific decision this pipeline is bad at. The
scorer reads a transcript and rates hook, humour and value; it has no notion
that a clip is about a gun, or that a joke about beating people up reads as
incitement once it is forty seconds long with no context around it. Both
came up on real material within one video: three of the highest-scored clips
were a Russian roulette bit, and one title generated was "Westworld incites
you to beat up people with glasses".

Posted to accounts a few days old, that is not a taste question. TikTok
removes dangerous-acts content and strikes the account; a strike on a
two-day-old account is most of its future.

So: a clip whose own words trip one of these markers is held rather than
posted. Held, not discarded — it is rendered, it is reported, and a person
can publish it in one command after looking at it. The cost of holding
something publishable is a clip that goes out an hour late. The cost of
publishing something that should have been held is the account.

Deliberately narrow. This is not a content-policy engine and cannot be one:
it is a short list of the things that actually cost an account, matched
against what is said in the clip, tuned to hold rarely.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Each entry: why it is held, and the phrasings that trip it. French first —
# that is what this is pointed at — with the English equivalents that show
# up in gaming speech either way.
#
# Anchored on phrases rather than single words wherever a single word would
# be ordinary speech. "arme" alone is a video game every evening; "roulette
# russe" is not.
RISK_MARKERS: dict[str, list[str]] = {
    "dangerous act": [
        r"roulette\s+russe", r"russian\s+roulette",
        r"jeu\s+du\s+foulard", r"choking\s+game",
        r"\bpendaison\b", r"se\s+pendre",
    ],
    "self-harm": [
        r"\bsuicid", r"me\s+suicider", r"se\s+suicider",
        r"me\s+tuer\b", r"kill\s+myself",
        r"s[ce]arification", r"me\s+couper\s+les\s+veines",
    ],
    "incitement": [
        r"tabasse[rz]?\s+(les|des|le|la)", r"casser\s+la\s+gueule\s+(à|a|aux)",
        r"faut\s+les?\s+(tuer|frapper|tabasser)",
        r"beat\s+up\s+(the|them)", r"go\s+punch",
    ],
    "weapon threat": [
        r"je\s+(te|vous)\s+(tire|bute|descends)\b",
        r"balle\s+dans\s+la\s+t[êe]te",
        r"\bflingue\b.{0,20}\b(sur|contre)\b",
    ],
    "slur": [
        # Held for a person to look at, not asserted to be an insult: these
        # are also used reclaimed and in quotation.
        r"\bp[ée]d[ée]s?\b", r"\bn[èe]gres?\b", r"\bbougnoule",
        r"\bsale\s+(juif|arabe|noir|blanc)",
    ],
}

_COMPILED = {
    reason: [re.compile(p, re.IGNORECASE) for p in patterns]
    for reason, patterns in RISK_MARKERS.items()
}


@dataclass
class Review:
    """Whether a clip may be posted without a person seeing it first."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"ok": self.ok, "reasons": self.reasons, "evidence": self.evidence}


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def check(*parts: str) -> Review:
    """Judge a clip from everything that would be published about it.

    Takes the transcript, the title and the description together: a title
    can be the problem even when the transcript is fine, which is exactly
    what happened with the "beat up people with glasses" headline.
    """
    haystack = _fold(" ".join(p for p in parts if p))
    reasons: list[str] = []
    evidence: list[str] = []
    for reason, patterns in _COMPILED.items():
        for pattern in patterns:
            match = pattern.search(haystack)
            if not match:
                continue
            if reason not in reasons:
                reasons.append(reason)
            start = max(0, match.start() - 40)
            evidence.append("…" + haystack[start : match.end() + 40].strip() + "…")
            break
    return Review(ok=not reasons, reasons=reasons, evidence=evidence[:4])
