"""Who a title is allowed to name.

The failure this prevents is not a crash. It is a public post stating that
a named real person did something the video does not show them doing.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.sources import naming

# Verbatim from the page this was built against.
TITLE = "QUI SUBIRA LA PIRE ÉPREUVE (ft Maxime Biaggi, Gotaga & Billy)"
DESCRIPTION = """Voir conditions générales. L'offre se termine le 30/09/2026.

Avec :
Billy
Gotaga
Maxime Biaggi

Un jeu créé par Squeezie & Théodore Bonnet

Réalisé par Théodore Bonnet

1ère Assistante réalisatrice : Yasmina Chambenoit
Directeur photo : Matthieu Menant
"""


# --- reading the page ----------------------------------------------------

def test_the_cast_comes_off_the_real_page():
    assert naming.cast(TITLE, DESCRIPTION, "Squeezie") == [
        "Billy", "Gotaga", "Maxime Biaggi", "Squeezie"
    ]


def test_the_crew_is_not_the_cast():
    """A director is named in the description and is not in the video.

    Attributing an on-screen action to the camera operator is exactly the
    kind of confident wrongness this has to avoid.
    """
    found = naming.cast(TITLE, DESCRIPTION, "Squeezie")
    for absent in ("Théodore Bonnet", "Yasmina Chambenoit", "Matthieu Menant"):
        assert absent not in found


def test_a_feat_fragment_alone_is_enough():
    assert naming.from_title("Le truc (ft Maxime Biaggi, Gotaga & Billy)") == [
        "Maxime Biaggi", "Gotaga", "Billy"
    ]


def test_avec_mid_sentence_is_a_preposition_not_a_heading():
    text = "On a tourné avec beaucoup de plaisir.\nMerci à tous."
    assert naming.from_description(text) == []


def test_a_page_with_nothing_useful_yields_nothing():
    assert naming.cast("Une vidéo", "Pas de générique ici.", "") == []


def test_a_channel_that_is_not_a_person_is_not_offered_as_one():
    assert "Team Cro√ªton 42" not in naming.cast("t", "", "Team Cro√ªton 42")


# --- who is in this clip -------------------------------------------------

def test_a_name_spoken_in_the_clip_is_available():
    cast = ["Billy", "Gotaga", "Maxime Biaggi"]
    assert naming.mentioned_in("Billy ! Il est mort en tombant !", cast) == ["Billy"]


def test_a_shortened_name_resolves_to_the_full_one():
    """People say "Max", credits say "Maxime Biaggi"."""
    cast = ["Billy", "Maxime Biaggi"]
    assert naming.mentioned_in("le pauvre Max, il croit s'en sortir", cast) == [
        "Maxime Biaggi"
    ]


def test_a_surname_alone_resolves_too():
    assert naming.mentioned_in("Biaggi arrive", ["Maxime Biaggi"]) == ["Maxime Biaggi"]


def test_accents_and_case_do_not_matter():
    # Whisper writes the same name both ways within one transcript.
    assert naming.mentioned_in("theodore est la", ["Théodore"]) == ["Théodore"]


def test_a_clip_naming_nobody_offers_nobody():
    cast = ["Billy", "Gotaga", "Maxime Biaggi"]
    assert naming.mentioned_in("Tu connais la roulette russe ?", cast) == []


def test_a_name_inside_another_word_does_not_count():
    assert naming.mentioned_in("le billet est sur la table", ["Billy"]) == []


# --- the guard -----------------------------------------------------------

def test_a_title_naming_someone_the_clip_does_not_is_dropped():
    """The prompt forbids guessing; this is the check that it obeyed.

    Naming Gotaga for something Maxime did is a false statement about a
    real person, published — not a style problem.
    """
    cast = ["Billy", "Gotaga", "Maxime Biaggi"]
    assert naming.strip_unsupported(
        "Gotaga joue à la roulette russe", ["Maxime Biaggi"], cast
    ) == ""


def test_a_title_naming_the_supported_person_survives():
    cast = ["Billy", "Gotaga", "Maxime Biaggi"]
    title = "Maxime Biaggi joue à la roulette russe"
    assert naming.strip_unsupported(title, ["Maxime Biaggi"], cast) == title


def test_a_title_naming_nobody_survives():
    cast = ["Billy", "Gotaga"]
    title = "Une roulette russe qui tourne mal"
    assert naming.strip_unsupported(title, [], cast) == title


def test_the_guard_matches_a_shortened_name_too():
    cast = ["Billy", "Maxime Biaggi"]
    assert naming.strip_unsupported("Max tombe de l'escabeau", ["Billy"], cast) == ""


def test_an_empty_title_stays_empty():
    assert naming.strip_unsupported("", ["Billy"], ["Billy"]) == ""


# --- the emoji set -------------------------------------------------------

def test_the_prompt_offers_a_closed_set_of_emoji():
    """Left open, a model reaches for a coffin on a clip about a staged
    death, and the post reads as a real event."""
    from publikclip_pipeline.scoring import rubric

    assert len(rubric.TITLE_EMOJI) >= 5
    prompt = rubric.headline_prompt("x", {"duration": 30})
    assert all(e in prompt for e in rubric.TITLE_EMOJI)
    assert "exactly ONE emoji" in prompt


def test_the_cast_block_appears_only_when_there_is_a_cast():
    from publikclip_pipeline.scoring import rubric

    assert "People in this video" not in rubric.headline_prompt("x", {"duration": 30})
    with_cast = rubric.headline_prompt(
        "x", {"duration": 30, "cast": ["Billy"], "named": ["Billy"]}
    )
    assert "People in this video: Billy" in with_cast
    assert "Named or addressed in THIS clip: Billy" in with_cast


def test_a_clip_naming_nobody_is_told_so_explicitly():
    from publikclip_pipeline.scoring import rubric

    prompt = rubric.headline_prompt(
        "x", {"duration": 30, "cast": ["Billy", "Gotaga"], "named": []}
    )
    assert "Nobody is named in this clip" in prompt
