"""Copy in the wrong language.

The prompt tells the model to answer in the transcript's language. A French
clip went out with an English caption anyway — "The streamer panics,
mentions going to the hospital" — because asking is not a guarantee. Every
string below is one that was actually published or one it must not catch.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.sources import language


# --- the captions that went out ------------------------------------------

@pytest.mark.parametrize("text", [
    "The streamer panics, mentions going to the hospital and asks for donations.",
    "The streamer thanks donors for 100€ gifts, then keeps playing.",
    "A wild moment when the player gets a clutch and everyone goes crazy.",
])
def test_english_copy_is_recognised(text):
    assert language.language_of(text) == "en"


@pytest.mark.parametrize("text", [
    "Elle ne s'appelle pas Lisa, le moment gênant d'un mariage français.",
    "Impro freestyle en français avec Billy, Gotaga et leurs potes.",
    "Il promet d'atomiser son rival en clash sur la prod.",
])
def test_french_copy_is_recognised(text):
    assert language.language_of(text) == "fr"


def test_a_french_sentence_quoting_english_is_still_french():
    """Streamers say "let's go" and "one shot" in French sentences; a guard
    that reads those as English deletes good copy."""
    assert language.language_of(
        "Le streamer part en live et fait un one shot, let's go les gars !"
    ) == "fr"


# --- when it must not answer ---------------------------------------------

@pytest.mark.parametrize("text", [
    "292€ en 15 minutes pour 10 000 balles !",
    "60 000 euros en une fois",
    "Gotaga le bulldozer court enfin",
    "",
    "🔥🔥🔥",
])
def test_too_little_to_call_returns_nothing(text):
    """A five-word title often cannot be classified, and a wrong answer here
    silently deletes a good headline."""
    assert language.language_of(text) is None


# --- the guard -----------------------------------------------------------

FR_SOURCE = "allez les gars on lache rien faites du bruit c'est parti pour la suite"


def test_the_real_failure_is_caught():
    assert language.mismatched(
        "The streamer thanks donors for 100€ gifts, then keeps playing.", FR_SOURCE
    )


def test_copy_in_the_right_language_is_kept():
    assert not language.mismatched(
        "Le streamer remercie les donateurs puis reprend sa partie.", FR_SOURCE
    )


def test_silence_means_keep():
    """Dropping a caption is a real loss, so it happens on evidence, never
    on the absence of it."""
    assert not language.mismatched("Gotaga le bulldozer court enfin", FR_SOURCE)
    assert not language.mismatched("Un titre 😱", "")
    assert not language.mismatched("", FR_SOURCE)


def test_an_english_clip_may_have_english_copy():
    english_source = "so we are going to try this again and see what happens now"
    assert not language.mismatched(
        "The streamer tries again and it works.", english_source
    )
    assert language.mismatched(
        "Il réessaie une fois de plus et cette fois tout le monde applaudit.",
        english_source,
    )


def test_a_short_mismatch_slips_through_and_that_is_the_trade():
    """The limit, stated rather than hidden. Six words carry two function
    words, under the minimum, so the guard says nothing and the copy is
    kept. Lowering the bar to catch this would start deleting good French
    titles, which are short by design — a caption in the wrong language is
    embarrassing, a deleted headline costs the clip its hook."""
    english_source = "so we are going to try this again and see what happens now"
    assert language.language_of("Il réessaie et ça marche.") is None
    assert not language.mismatched("Il réessaie et ça marche.", english_source)
