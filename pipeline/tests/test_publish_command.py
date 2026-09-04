"""Publishing a job that was rendered by hand.

`run` produced clips and `auto` published them, with nothing between: a job
cut deliberately — the usual case when the source is a YouTube URL the
autopilot cannot reach — could not be posted at all.
"""

from __future__ import annotations

import json

import pytest

from publikclip_pipeline import cli
from publikclip_pipeline.autopilot import publish as publish_mod


def a_job(tmp_path, scores=(52.0, 44.0, 21.0)):
    d = tmp_path / "job"
    (d / "clips").mkdir(parents=True)
    outputs = []
    for i, score in enumerate(scores):
        path = d / "clips" / f"clip_{i:02d}.mp4"
        path.write_bytes(b"\x00" * 512)
        outputs.append({
            "clip": i,
            "path": str(path),
            "score": score,
            "best_platform": "reels",
            "duration": 30.0,
            "title": f"TITRE {i}",
            "description": f"description {i}",
            "hashtags": ["a", "b"],
        })
    (d / "render.json").write_text(
        json.dumps({"stage": "render", "schema_version": 2, "created_at": 0,
                    "data": {"outputs": outputs}}),
        encoding="utf-8",
    )
    (d / "score.json").write_text(
        json.dumps({"stage": "score", "schema_version": 1, "created_at": 0,
                    "data": {"clips": [
                        {"summary": f"resume {i}", "confidence": "third-party"}
                        for i, _ in enumerate(scores)
                    ]}}),
        encoding="utf-8",
    )
    return d


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path / "home"))


def run(argv):
    return cli.main(argv)


def test_a_hand_cut_job_can_be_published(tmp_path, capsys):
    d = a_job(tmp_path)
    code = run([
        "publish", "j1", "--dir", str(d), "--platforms", "tiktok",
        "--min-score", "40", "--clips", "3",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [c["clip"] for c in payload["selected"]] == [0, 1]
    assert payload["failures"] == 0


def test_the_floor_still_applies(tmp_path, capsys):
    d = a_job(tmp_path)
    code = run([
        "publish", "j1", "--dir", str(d), "--platforms", "tiktok",
        "--min-score", "50",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [c["clip"] for c in payload["selected"]] == [0]


def test_nothing_above_the_floor_is_not_a_failure(tmp_path, capsys):
    d = a_job(tmp_path, scores=(10.0, 12.0))
    assert run(["publish", "j1", "--dir", str(d), "--min-score", "40"]) == 0
    assert "scored at or above" in capsys.readouterr().err


def test_a_missing_job_says_so_rather_than_traversing_nothing(tmp_path, capsys):
    code = run(["publish", "nope", "--dir", str(tmp_path / "absent")])
    assert code == 2
    assert "No job" in capsys.readouterr().err


def test_dry_run_is_the_default(tmp_path, capsys):
    """The same default as everywhere else: posting is opt-in."""
    d = a_job(tmp_path)
    run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok"])
    payload = json.loads(capsys.readouterr().out)
    assert all(p["dry_run"] for p in payload["published"])


def test_a_bad_platform_name_is_refused_before_any_posting(tmp_path, capsys):
    d = a_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "reels"]) == 1
    assert "instagram" in capsys.readouterr().err


def test_the_generated_copy_reaches_the_publisher(tmp_path, capsys):
    d = a_job(tmp_path)
    run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok"])
    payload = json.loads(capsys.readouterr().out)
    first = payload["selected"][0]
    assert first["title"] == "TITRE 0"
    assert first["description"] == "description 0"
    assert first["hashtags"] == ["a", "b"]


def test_a_clip_already_posted_is_not_posted_twice(tmp_path, capsys, monkeypatch):
    """The ledger is shared with the autopilot, so re-running this after a
    burst must not duplicate anything."""
    d = a_job(tmp_path)
    seen = []
    monkeypatch.setattr(
        publish_mod, "already_posted",
        lambda clip, platform: seen.append((clip.clip, platform)) or clip.clip == 0,
    )
    run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok", "--min-score", "40"])
    payload = json.loads(capsys.readouterr().out)
    assert [p["clip"] for p in payload["published"]] == [1]
    assert (0, "tiktok") in seen


# --- clips that were asked for, not ranked -------------------------------
#
# The fourth place the same principle had to be learned. --cut reaches the
# candidate stage, survives the scorer's finalist limit, reaches the render
# — and was then dropped here, by a score floor, having been made.

def a_dictated_job(tmp_path, scores=(52.0, 44.0, 21.0), dictated=(2,)):
    d = a_job(tmp_path, scores)
    doc = json.loads((d / "score.json").read_text(encoding="utf-8"))
    for i in dictated:
        doc["data"]["clips"][i]["dictated"] = True
    (d / "score.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


def test_a_dictated_clip_is_published_below_the_floor(tmp_path, capsys):
    """21.0 against a floor of 40. Somebody watched the video and named
    these bounds; the floor is there to filter what nobody chose."""
    d = a_dictated_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--min-score", "40", "--clips", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 2 in [c["clip"] for c in payload["selected"]]


def test_a_dictated_clip_takes_the_place_of_a_better_scoring_one(tmp_path, capsys):
    """--clips 1 means one post, not one plus however many were dictated.
    The scorer reads its own limit the same way, and two meanings for the
    same number is worse than either meaning."""
    d = a_dictated_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--min-score", "0", "--clips", "1"]) == 0
    picked = [c["clip"] for c in json.loads(capsys.readouterr().out)["selected"]]
    assert picked == [2], "the dictated clip lost its place to a better score"


def test_the_take_still_buys_merit_clips_alongside(tmp_path, capsys):
    d = a_dictated_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--min-score", "0", "--clips", "2"]) == 0
    picked = [c["clip"] for c in json.loads(capsys.readouterr().out)["selected"]]
    assert picked == [2, 0], "the dictated one, then the best of the rest"


def test_dictated_clips_come_first(tmp_path, capsys):
    d = a_dictated_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--min-score", "0"]) == 0
    picked = [c["clip"] for c in json.loads(capsys.readouterr().out)["selected"]]
    assert picked[0] == 2


# --- naming exact clips --------------------------------------------------

def test_only_posts_exactly_what_it_names(tmp_path, capsys):
    d = a_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--only", "2,0", "--min-score", "40", "--clips", "1"]) == 0
    picked = [c["clip"] for c in json.loads(capsys.readouterr().out)["selected"]]
    assert picked == [2, 0], "the order asked for is the order"


def test_only_overrides_the_floor(tmp_path, capsys):
    d = a_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--only", "2", "--min-score", "50"]) == 0
    picked = [c["clip"] for c in json.loads(capsys.readouterr().out)["selected"]]
    assert picked == [2]


def test_only_ignores_a_clip_that_does_not_exist(tmp_path, capsys):
    d = a_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--only", "0,99"]) == 0
    picked = [c["clip"] for c in json.loads(capsys.readouterr().out)["selected"]]
    assert picked == [0]


def test_only_rejects_something_that_is_not_a_clip_number(tmp_path, capsys):
    d = a_job(tmp_path)
    assert run(["publish", "j1", "--dir", str(d), "--platforms", "tiktok",
                "--only", "le premier"]) == 1
