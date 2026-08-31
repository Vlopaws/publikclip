"""Settings must survive the trip through a job's settings.json.

The whole point of snapshotting settings into the job directory is that a
resumed job never silently picks up changed defaults. A field that does not
survive serialization breaks exactly that guarantee, and does it quietly:
the caller sets it, every line of code that reads it looks correct, and the
run behaves as though nobody had asked.

That is not hypothetical. `max_finalists` was added to the dataclass, set by
the autopilot, and dropped by a hand-written to_json — so a run asked for
six clips and rendered twelve.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from publikclip_pipeline import config


def test_every_field_survives_a_round_trip():
    """The guard the hand-written version could not give: this fails the
    moment a new setting is added without thinking about persistence."""
    settings = config.Settings()
    restored = config.Settings.from_json(json.loads(json.dumps(settings.to_json())))
    for f in fields(config.Settings):
        assert getattr(restored, f.name) == getattr(settings, f.name), f.name


def test_a_non_default_value_survives_rather_than_reverting():
    settings = config.Settings()
    settings.max_finalists = 6
    settings.llm_mode = "groq"
    settings.lufs_target = -16.0
    restored = config.Settings.from_json(json.loads(json.dumps(settings.to_json())))
    assert restored.max_finalists == 6
    assert restored.llm_mode == "groq"
    assert restored.lufs_target == -16.0


def test_nested_camera_settings_survive():
    settings = config.Settings()
    settings.camera.framing = "wide"
    settings.camera.punch_in = False
    restored = config.Settings.from_json(json.loads(json.dumps(settings.to_json())))
    assert restored.camera.framing == "wide"
    assert restored.camera.punch_in is False
    assert isinstance(restored.camera, config.CameraSettings)


def test_every_field_reaches_the_json():
    names = {f.name for f in fields(config.Settings)}
    assert set(config.Settings().to_json()) == names


def test_a_snapshot_written_before_a_field_existed_still_loads():
    # A job directory outlives the build that wrote it.
    old = {"llm_mode": "groq", "camera": {"speaker_change": "cut"}}
    restored = config.Settings.from_json(old)
    assert restored.llm_mode == "groq"
    assert restored.max_finalists is None  # the default, not a crash


def test_a_snapshot_from_a_newer_build_is_not_fatal():
    ahead = config.Settings().to_json()
    ahead["some_setting_from_the_future"] = 42
    ahead["camera"]["a_new_camera_knob"] = True
    restored = config.Settings.from_json(ahead)
    assert restored.llm_mode == config.Settings().llm_mode


def test_the_snapshot_is_plain_json():
    # It is written to disk and read back by another process; anything that
    # needs a custom encoder does not belong in it.
    json.dumps(config.Settings().to_json())


def test_the_autopilots_cap_reaches_a_job(tmp_path, monkeypatch):
    """End to end through the queue, which is where it actually broke."""
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    from publikclip_pipeline.jobs import queue

    settings = config.Settings()
    settings.max_finalists = 6
    job = queue.create_job("url", "https://example.com/v", json.dumps(settings.to_json()))

    on_disk = json.loads((job.dir / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["max_finalists"] == 6
    assert config.Settings.from_json(json.loads(job.settings_json)).max_finalists == 6
