"""A cast member's name, as the transcriber heard it.

The cast list comes from the page — "(ft Maxime Biaggi, Gotaga & Billy)" —
and is the only ground truth for how a name is spelt. A transcript is not.
Measured on the run that exposed this: "Gotha" appeared 23 times in the
transcript and "Gotaga" 13, so a title written from that transcript said
"Gotha" without any model inventing anything.

Every string here is one that actually occurred or one that must not be
touched.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.sources import naming

CAST = ["Maxime Biaggi", "Gotaga", "Billy", "Squeezie"]


# --- the names that were wrong ------------------------------------------

def test_the_name_the_transcriber_mangled_is_respelt():
    assert naming.correct_spelling("Impro freestyle avec Billy et Gotha 😱", CAST) == \
        "Impro freestyle avec Billy et Gotaga 😱"


def test_the_uploader_too():
    assert naming.correct_spelling("Il se prend pour Squeezil rappeur 🤯", CAST) == \
        "Il se prend pour Squeezie rappeur 🤯"


def test_a_name_already_right_is_left_alone():
    for title in ("Gotaga le bulldozer court enfin 😱",
                  "Maxime rentre en patate",
                  "Billy balance des rimes"):
        assert naming.correct_spelling(title, CAST) == title


# --- the names that must survive untouched -------------------------------

def test_someone_not_in_the_cast_is_not_renamed_into_it():
    """Dimé and MCBADG are spoken in the clip and are nobody on the list.
    Guessing that they are really a cast member would be the fabrication
    this function exists to undo."""
    for title in ("Gotaga se fait insulter par Dimé en live 😱",
                  "MCBADG répond avec un flow explosif",
                  "tout le monde crie Jimmy !",
                  "Elle ne s'appelle pas Lisa 😱"):
        assert naming.correct_spelling(title, CAST) == title


def test_a_rhyme_is_not_a_mishearing():
    """0.667 against Gotaga — below the gate, and it is a place."""
    assert naming.correct_spelling("Le chevalier de Gotham", ["Gotaga"]) == \
        "Le chevalier de Gotham"


def test_a_diminutive_is_not_corrected_to_the_full_name():
    """"Max" is what people call him; expanding it would put words in a
    title that nobody said."""
    assert naming.correct_spelling("Allez Max !", CAST) == "Allez Max !"


def test_lower_case_words_are_never_touched():
    """A name sits in proper-noun position. A lower-case word that happens
    to resemble one is a word."""
    assert naming.correct_spelling("il faut gotha ici", ["Gotaga"]) == \
        "il faut gotha ici"


def test_an_ordinary_capitalised_word_is_safe():
    assert naming.correct_spelling("Bien joué les gars", CAST) == "Bien joué les gars"


# --- edges ---------------------------------------------------------------

def test_no_cast_means_no_correction():
    """With nothing to check against, changing a name would be a guess."""
    assert naming.correct_spelling("avec Gotha", []) == "avec Gotha"


def test_an_empty_title_is_returned_as_given():
    assert naming.correct_spelling("", CAST) == ""
    assert naming.correct_spelling(None, CAST) is None


def test_a_two_word_cast_name_matches_on_its_first_word():
    assert naming.correct_spelling("Maximme assure", ["Maxime Biaggi"]) == \
        "Maxime assure"


def test_every_occurrence_is_corrected():
    assert naming.correct_spelling("Gotha contre Gotha", ["Gotaga"]) == \
        "Gotaga contre Gotaga"


def test_the_gate_sits_between_the_measured_cases():
    """The real mishearing and the nearest thing that must not move."""
    assert naming._misheard_as("Gotha", "Gotaga")
    assert not naming._misheard_as("Gotham", "Gotaga")
