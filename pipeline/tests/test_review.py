"""What an unattended run refuses to publish by itself.

The operator asked for autonomous public posting, which removes the person
who was the last check. It also removes them from a decision the scorer is
bad at: it rates hook, humour and value from a transcript and has no notion
that a clip is about a gun.

Both cases here are real, from one video: three of the highest-scored clips
were a Russian roulette bit, and one generated title was "Westworld incites
you to beat up people with glasses". On accounts a few days old that is not
a taste question — TikTok strikes the account.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.autopilot import review


# --- what gets held ------------------------------------------------------

def test_the_russian_roulette_clip_is_held():
    """Verbatim from the clip that scored 44.9, the second best in the video."""
    said = (
        "Tu connais la roulette russe ? On va laisser le hasard décider à ta "
        "place. Je meurs ou tu meurs ? Dans les deux cas, ta dette est remboursée."
    )
    verdict = review.check(said, "", "")
    assert not verdict.ok
    assert "dangerous act" in verdict.reasons
    assert verdict.evidence


def test_a_title_can_be_the_problem_on_its_own():
    """The transcript was ordinary; the generated headline was not."""
    verdict = review.check(
        "on parle de lunettes et de jeux vidéo",
        "Westworld incite à tabasser les binoclards",
        "",
    )
    assert not verdict.ok
    assert "incitement" in verdict.reasons


def test_the_description_is_judged_too():
    verdict = review.check("rien de spécial", "", "il dit qu'il veut se suicider")
    assert not verdict.ok
    assert "self-harm" in verdict.reasons


def test_a_direct_threat_is_held():
    assert not review.check("je te bute si tu bouges", "", "").ok


def test_every_reason_that_applies_is_reported():
    verdict = review.check("roulette russe et je te bute", "", "")
    assert set(verdict.reasons) >= {"dangerous act", "weapon threat"}


def test_accents_and_case_do_not_let_it_through():
    assert not review.check("ROULETTE RUSSE", "", "").ok
    assert not review.check("Roulette Rüsse".replace("ü", "u"), "", "").ok


# --- what goes through ---------------------------------------------------

def test_an_ordinary_gaming_clip_passes():
    said = (
        "Premier kill de la partie, il arrive par derrière, on s'arrête pas "
        "là les gars, allez on relance"
    )
    assert review.check(said, "Premier kill de Simov en direct", "").ok


def test_a_weapon_word_alone_is_not_enough():
    """"arme" and "tirer" are every shooter, every evening. Holding on
    those would hold everything and the guard would be turned off."""
    assert review.check("je tire sur le boss avec mon arme", "", "").ok
    assert review.check("il a un flingue dans le jeu", "", "").ok


def test_an_interview_passes():
    said = (
        "La question c'est de savoir si les métadonnées suffisent à "
        "identifier quelqu'un, et la réponse est oui."
    )
    assert review.check(said, "Les métadonnées suffisent à vous identifier", "").ok


def test_nothing_at_all_passes():
    assert review.check("", "", "").ok
    assert review.check(None, None, None).ok


# --- the shape of the answer --------------------------------------------

def test_a_held_clip_says_why_and_shows_the_words():
    verdict = review.check("on joue à la roulette russe ce soir", "", "")
    assert verdict.reasons == ["dangerous act"]
    assert any("roulette" in e for e in verdict.evidence)
    assert verdict.to_json()["ok"] is False


def test_evidence_is_bounded():
    said = " ".join(["roulette russe je te bute suicide tabasser les gens"] * 20)
    assert len(review.check(said, "", "").evidence) <= 4


# --- how it is used ------------------------------------------------------

def test_the_autopilot_holds_rather_than_publishes(monkeypatch, tmp_path):
    """Held, not discarded: the clip is rendered and one command from going
    out. The cost of holding something publishable is an hour's delay; the
    cost of publishing something that should have been held is the account.
    """
    from publikclip_pipeline.autopilot import runner

    source = __import__(
        "publikclip_pipeline.sources.common", fromlist=["SourceItem"]
    ).SourceItem(
        id="v1", url="https://x/v1", title="t", source="twitch", duration_sec=40.0
    )
    clip = __import__(
        "publikclip_pipeline.autopilot.select", fromlist=["SelectedClip"]
    ).SelectedClip(
        job_id="j1", clip=0, path=tmp_path / "c.mp4", score=44.9,
        best_platform="reels", duration=40.0, summary="s",
        confidence="third-party",
        transcript="Tu connais la roulette russe ?",
    )

    published = []

    class Publisher:
        name = "fake"
        visibility = "public"

        def check_ready(self, platforms):
            pass

        def publish(self, c, platform):
            published.append((c.clip, platform))
            raise AssertionError("published a clip that should have been held")

    monkeypatch.setattr(runner, "_process", lambda *a, **k: "j1")
    monkeypatch.setattr(runner, "select", lambda *a, **k: [clip])
    monkeypatch.setattr(runner.queue, "get_job", lambda jid: type("J", (), {"dir": tmp_path})())

    report = runner.run(
        [source], publisher=Publisher(), platforms=["tiktok"], skip_seen=False
    )
    assert published == []
    assert report.to_json()["clips_held"] == 1
    held = report.outcomes[0].held[0]
    assert "dangerous act" in held["reasons"]
