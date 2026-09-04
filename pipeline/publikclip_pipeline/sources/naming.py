"""Who is in this video, and who is in this clip.

A title that says "a hostile bar patron suggests Russian roulette" is a
description. One that says "Maxime Biaggi joue à la roulette russe" is a
clip. The difference is a name, and the name is already in the page — it
just never reached the part that writes the copy.

Two sources, both free and both already fetched:

  - the video title, which carries "(ft Maxime Biaggi, Gotaga & Billy)"
  - the description, which usually carries an explicit cast block:

        Avec :
        Billy
        Gotaga
        Maxime Biaggi

Deliberately NOT face or voice recognition. Identifying a specific creator
from pixels needs a labelled database of that creator, which does not exist
here and would be wrong the first time someone new appeared. The metadata
says who is present; the transcript says who is named in this moment; the
intersection is what a title may claim.

That intersection is the whole safety property. These are real people and
the post is public, so attributing an action to the wrong one is a factual
claim about somebody that nothing in the video supports. A name is allowed
in a title only when the clip's own words contain it.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
import unicodedata

# Lines that introduce a cast list, across the phrasings French and English
# channels actually use. Matched at the start of a line only: "avec" in the
# middle of a sentence is a preposition, not a heading.
_CAST_HEADINGS = re.compile(
    r"^\s*(avec|with|cast|featuring|invit[ée]s?|participants?)\s*[:：]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# "(ft Maxime Biaggi, Gotaga & Billy)" and its spellings.
_FEAT = re.compile(
    r"\b(?:ft|feat|featuring|avec|with)\.?\s+(?P<names>[^()\[\]|·—]+)",
    re.IGNORECASE,
)

# Crew credits are names too, and they are not in the video. A line with a
# role before the colon is somebody behind the camera.
_CREW_LINE = re.compile(
    r"(r[ée]alis|assistant|photo|cadreu|monteur|montage|musique|son|"
    r"producteur|production|graphis|d[ée]cor|maquill|r[ée]gie|scripte|"
    r"[ée]tallonnage|mixage|direct)",
    re.IGNORECASE,
)

# A plausible display name: one to three capitalised words, no digits.
_NAME = re.compile(r"^[A-ZÀ-ÖØ-Þ][\w'’\-]+(?:\s+[A-ZÀ-ÖØ-Þ0-9][\w'’\-]*){0,2}$")

_SPLIT = re.compile(r"\s*(?:,|&|\bet\b|\band\b|\+|/|\|)\s*", re.IGNORECASE)

# Long enough to be a name, short enough not to be a sentence.
MAX_NAME_WORDS = 3
MAX_CAST = 12


def _clean(text: str) -> str:
    return " ".join((text or "").replace("​", " ").split()).strip(" -–—•·:")


def _plausible(name: str) -> bool:
    name = _clean(name)
    if not name or len(name) < 2 or len(name.split()) > MAX_NAME_WORDS:
        return False
    if any(ch.isdigit() for ch in name):
        return False
    return bool(_NAME.match(name))


def _fold(text: str) -> str:
    """Accent- and case-insensitive form, for matching a name against speech.

    Whisper writes "Theodore" as often as "Théodore", and a transcript is
    lower-cased mid-sentence as readily as not.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def from_title(title: str) -> list[str]:
    """Names from a "(ft A, B & C)" fragment in the title."""
    found: list[str] = []
    for match in _FEAT.finditer(title or ""):
        for part in _SPLIT.split(match.group("names")):
            if _plausible(part):
                found.append(_clean(part))
    return found


def from_description(description: str) -> list[str]:
    """Names from an explicit cast block.

    Only the run of lines immediately under the heading, stopping at the
    first blank line or crew credit. A description continues into the whole
    crew list, and a director is not in the video.
    """
    if not description:
        return []
    found: list[str] = []
    lines = description.splitlines()
    for i, line in enumerate(lines):
        if not _CAST_HEADINGS.match(line):
            continue
        for candidate in lines[i + 1 :]:
            stripped = candidate.strip()
            if not stripped:
                break
            if _CREW_LINE.search(stripped) or ":" in stripped:
                break
            for part in _SPLIT.split(stripped):
                if _plausible(part):
                    found.append(_clean(part))
    return found


def cast(title: str = "", description: str = "", uploader: str = "") -> list[str]:
    """Everyone plausibly on camera, most explicit source first.

    The uploader comes last and only if it looks like a person's name: the
    channel owner is usually present, but "Team Croûton" is not a person to
    attribute an action to.
    """
    names: list[str] = []
    seen: set[str] = set()
    for source in (from_description(description), from_title(title), [uploader]):
        for name in source:
            if not _plausible(name):
                continue
            key = _fold(name)
            if key in seen:
                continue
            seen.add(key)
            names.append(_clean(name))
    return names[:MAX_CAST]


# Speech shortens names. People say "Max" for "Maxime" and "Gota" for
# "Gotaga", so a spoken token counts if it is a prefix of a cast name part.
#
# Three characters, measured rather than picked: on the video this was
# built against, "max" occurs 23 times and never once as the French
# intensifier "au max" — the false friend that makes this rule risky in
# general did not appear at all. Two characters would match far too much.
#
# The rule is a hint, not an authority. It says who a title MAY name; the
# model still has the transcript and is told to omit the name when the
# clip does not make the attribution obvious.
MIN_SPOKEN_PREFIX = 3


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _fold(text or "")))


def _names_a_person(tokens: set[str], name: str) -> bool:
    """Does this bag of words refer to `name`?

    One rule, used by both the detector and the guard. Splitting them was a
    real hole: "Max" was enough to FIND Maxime Biaggi in a transcript but
    not enough to BLOCK a title that named him, so a wrong attribution in
    its shortened form went straight through.
    """
    parts = [p for p in _fold(name).split() if len(p) >= MIN_SPOKEN_PREFIX]
    return any(
        token == part
        or (len(token) >= MIN_SPOKEN_PREFIX and part.startswith(token))
        for part in parts
        for token in tokens
    )


def mentioned_in(transcript: str, names: list[str]) -> list[str]:
    """Which of `names` this clip's own words plausibly contain.

    Matches on any word of the name, so "Biaggi" resolves to "Maxime
    Biaggi" — people are listed by two names in credits and addressed by
    one in speech. Word-boundary anchored, so "Billy" does not match inside
    another word, and prefix-tolerant, so "Max" reaches "Maxime".

    This is the list a title may draw on. Everything outside it would be
    the model deciding who did something, about real people, on a public
    post.
    """
    tokens = _tokens(transcript)
    return [name for name in names if _names_a_person(tokens, name)]


def strip_unsupported(title: str, allowed: list[str], full_cast: list[str]) -> str:
    """Remove a name the clip does not support from a generated title.

    The prompt forbids guessing, and this is the check that the prompt was
    obeyed. A model naming the wrong cast member is not a style problem: it
    is a false statement about a real person, published.
    """
    if not title:
        return title
    tokens = _tokens(title)
    allowed_set = {_fold(a) for a in allowed}
    for name in full_cast:
        if _fold(name) in allowed_set:
            continue
        if _names_a_person(tokens, name):
            return ""
    return title


# How close a word must be to a cast member's name to be that name misspelt.
#
# Measured on the run that produced it: the transcript said "Gotha" 23 times
# and "Gotaga" 13, so the model wrote "Gotha" in a title by faithfully
# copying what it was given -- the fabrication was the transcriber's, not
# the model's. Gotha scores 0.727 against Gotaga; the nearest thing that
# must NOT be corrected (Gotaga against an unrelated "Gotham") scores 0.667.
# The gate sits between them, with the shared-prefix rule below carrying
# most of the weight.
NAME_MISHEARD_RATIO = 0.70

# A mishearing keeps the start of a name and mangles the rest -- "Gotaga"
# becomes "Gotha", never "Rotaga". Requiring three shared characters is what
# separates a misspelling from a different person who merely rhymes.
NAME_PREFIX = 3


def _misheard_as(word: str, name: str) -> bool:
    """Is `word` this name, misheard?"""
    a, b = _fold(word), _fold(name)
    if not a or not b or a == b:
        return False  # already correct, or nothing to compare
    if a[:NAME_PREFIX] != b[:NAME_PREFIX]:
        return False
    return SequenceMatcher(None, a, b).ratio() >= NAME_MISHEARD_RATIO


def correct_spelling(text: str, cast: list[str]) -> str:
    """Respell a cast member's name that the transcriber got wrong.

    The cast list comes from the page -- "(ft Maxime Biaggi, Gotaga &
    Billy)" -- and is the only ground truth for how a name is spelt. A
    transcript is not: ASR renders "Gotaga" as "Gotha" and "Squeezie" as
    "Squeezil", and anything written from that transcript inherits the
    error.

    Correcting is right where stripping would be wrong. The person really is
    in the video; only the spelling is not. A title that drops the name says
    less than one that spells it properly, and a viewer searching the name
    finds neither the misspelling nor the silence.
    """
    if not text or not cast:
        return text

    # Longest first, so "Maxime Biaggi" is considered before "Maxime".
    targets: list[str] = []
    for name in cast:
        targets.append(name)
        first = name.split()[0] if name.split() else ""
        if first and first != name:
            targets.append(first)
    targets.sort(key=len, reverse=True)

    def fix(match: re.Match) -> str:
        word = match.group(0)
        for name in targets:
            if " " in name:
                continue  # a single token cannot be a two-word name
            if _misheard_as(word, name):
                return name
        return word

    # Only capitalised words: a name sits in proper-noun position, and a
    # lower-case word that happens to resemble one is a word.
    return re.sub(r"\b[A-ZÀ-ÖØ-Þ][\w'’\-]+", fix, text)
