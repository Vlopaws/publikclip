"""Hosted transcription.

The word re-attachment gets the most attention: Groq returns words in one
flat list while everything downstream — captions, clip boundaries, speaker
assignment — reads them nested per segment. Getting that wrong produces
captions that drift out of sync rather than an error.
"""

import pytest

from publikclip_pipeline.asr import hosted, stage


def _seg(start, end, text="x"):
    return {"start": start, "end": end, "text": text}


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


# --- word re-attachment ---------------------------------------------------


def test_words_land_in_the_segment_that_contains_them():
    segments = [_seg(0.0, 2.0, "hello there"), _seg(2.0, 4.0, "second bit")]
    words = [_w("hello", 0.1, 0.5), _w("there", 0.6, 1.2),
             _w("second", 2.1, 2.6), _w("bit", 2.7, 3.1)]
    out = hosted._attach_words(segments, words)
    assert [[w["word"] for w in s["words"]] for s in out] == [
        ["hello", "there"], ["second", "bit"]
    ]


def test_a_word_on_the_boundary_goes_to_the_later_segment():
    """Segment ends are exclusive; a word starting exactly at the boundary
    belongs to what follows, or it would be counted twice."""
    segments = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
    out = hosted._attach_words(segments, [_w("edge", 2.0, 2.4)])
    assert [w["word"] for w in out[0]["words"]] == []
    assert [w["word"] for w in out[1]["words"]] == ["edge"]


def test_words_before_any_segment_are_dropped_not_misfiled():
    segments = [_seg(5.0, 6.0)]
    out = hosted._attach_words(segments, [_w("stray", 0.1, 0.4), _w("kept", 5.1, 5.4)])
    assert [w["word"] for w in out[0]["words"]] == ["kept"]


def test_a_gap_between_segments_does_not_shift_later_words():
    """A word in a silence gap must not be swept into the next segment."""
    segments = [_seg(0.0, 1.0), _seg(10.0, 11.0)]
    out = hosted._attach_words(segments, [_w("a", 0.1, 0.5), _w("noise", 4.0, 4.2), _w("b", 10.1, 10.5)])
    assert [w["word"] for w in out[0]["words"]] == ["a"]
    assert [w["word"] for w in out[1]["words"]] == ["b"]


def test_a_segment_with_no_words_still_appears():
    out = hosted._attach_words([_seg(0.0, 1.0, "music")], [])
    assert len(out) == 1 and out[0]["words"] == []


def test_the_downstream_shape_is_exactly_what_whisperx_produced():
    out = hosted._attach_words([_seg(0.0, 1.0, "  hi  ")], [_w("  hi  ", 0.1, 0.5)])
    seg = out[0]
    assert set(seg) == {"start", "end", "text", "words"}
    assert seg["text"] == "hi", "text is stripped like the local path"
    word = seg["words"][0]
    assert set(word) == {"word", "start", "end", "score"}
    assert word["word"] == "hi"
    assert word["score"] == 0.0, "no per-word confidence exists hosted"


def test_timestamps_are_rounded_like_the_local_path():
    out = hosted._attach_words([_seg(0.123456, 1.987654)], [_w("x", 0.200000, 0.555555)])
    assert out[0]["start"] == 0.123
    assert out[0]["words"][0]["end"] == 0.556


def test_a_word_starting_just_before_its_segment_is_kept():
    """Segment and word timestamps are rounded independently, so the first
    word of a segment often starts a few milliseconds early. Dropping it
    silently loses a word from the captions."""
    out = hosted._attach_words([_seg(1.000, 2.000)], [_w("early", 0.985, 1.300)])
    assert [w["word"] for w in out[0]["words"]] == ["early"]


def test_slack_does_not_steal_from_the_previous_segment():
    out = hosted._attach_words(
        [_seg(0.0, 1.0), _seg(1.0, 2.0)], [_w("mine", 0.97, 0.99)]
    )
    assert [w["word"] for w in out[0]["words"]] == ["mine"]
    assert out[1]["words"] == []


# --- backend selection ----------------------------------------------------


def test_hosted_is_preferred_when_a_key_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.delenv("PUBLIKCLIP_ASR_BACKEND", raising=False)
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_test")
    assert stage._choose_backend() == "groq"


def test_local_is_used_when_there_is_no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.delenv("PUBLIKCLIP_ASR_BACKEND", raising=False)
    monkeypatch.delenv(hosted.API_KEY_ENV, raising=False)
    assert stage._choose_backend() == "local"


def test_local_can_be_forced_even_with_a_key(monkeypatch, tmp_path):
    """Keeping audio on the machine has to stay possible."""
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_test")
    monkeypatch.setenv("PUBLIKCLIP_ASR_BACKEND", "local")
    assert stage._choose_backend() == "local"


def test_missing_key_names_where_to_get_one(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.delenv(hosted.API_KEY_ENV, raising=False)
    with pytest.raises(hosted.HostedAsrError) as err:
        hosted.transcribe(tmp_path / "nope.wav")
    assert "console.groq.com" in str(err.value)
    assert hosted.API_KEY_ENV in str(err.value)


# --- upload handling ------------------------------------------------------


def test_a_small_file_is_uploaded_as_is(monkeypatch, tmp_path):
    """Compression is for oversized audio only. Re-encoding a small file
    would cost time for nothing — and so would probing its duration, which
    is why neither subprocess may be reached on this path."""
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_test")
    audio = tmp_path / "audio16k.wav"
    audio.write_bytes(b"0" * 1024)
    called = []
    monkeypatch.setattr(
        hosted, "_extract_flac", lambda src, dest, s, l: called.append("flac") or src
    )
    monkeypatch.setattr(
        hosted, "_audio_duration", lambda p: called.append("probe") or 1.0
    )

    class _Res:
        status_code = 200

        @staticmethod
        def json():
            return {"language": "fr", "segments": [_seg(0.0, 1.0)], "words": [_w("a", 0.1, 0.4)]}

    monkeypatch.setattr(hosted.httpx, "post", lambda *a, **k: _Res())
    out = hosted.transcribe(audio)
    assert called == [], "a small file was needlessly re-encoded or probed"
    assert out["backend"] == "groq"
    assert out["word_count"] == 1


def test_an_oversized_file_is_compressed_first(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_test")
    audio = tmp_path / "audio16k.wav"
    audio.write_bytes(b"0" * (hosted.UPLOAD_LIMIT_BYTES + 1))
    flac = tmp_path / "audio16k.upload.flac"
    flac.write_bytes(b"0" * 1024)
    # Compression and probing are the two subprocess calls on this path.
    monkeypatch.setattr(hosted, "_audio_duration", lambda p: 1200.0)
    monkeypatch.setattr(hosted, "_extract_flac", lambda src, dest, s, l: flac)

    class _Res:
        status_code = 200

        @staticmethod
        def json():
            return {"segments": [_seg(0.0, 1.0)], "words": []}

    monkeypatch.setattr(hosted.httpx, "post", lambda *a, **k: _Res())
    assert hosted.transcribe(audio)["backend"] == "groq"


def test_still_oversized_after_compression_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_test")
    audio = tmp_path / "audio16k.wav"
    audio.write_bytes(b"0" * (hosted.UPLOAD_LIMIT_BYTES + 1))
    # Compression that gains nothing: every part still overruns the limit,
    # and the operator is told what to do about it rather than watching a
    # 413 come back from the server.
    monkeypatch.setattr(hosted, "_audio_duration", lambda p: 60.0)
    monkeypatch.setattr(hosted, "_extract_flac", lambda src, dest, s, l: audio)
    with pytest.raises(hosted.HostedAsrError) as err:
        hosted.transcribe(audio)
    assert "PUBLIKCLIP_ASR_BACKEND=local" in str(err.value)


def test_a_rejected_key_is_reported_plainly(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_bad")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"0" * 16)

    class _Res:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr(hosted.httpx, "post", lambda *a, **k: _Res())
    with pytest.raises(hosted.HostedAsrError) as err:
        hosted.transcribe(audio)
    assert "rejected the API key" in str(err.value)


def test_silent_audio_is_an_error_not_an_empty_success(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(hosted.API_KEY_ENV, "gsk_test")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"0" * 16)

    class _Res:
        status_code = 200

        @staticmethod
        def json():
            return {"segments": [], "words": []}

    monkeypatch.setattr(hosted.httpx, "post", lambda *a, **k: _Res())
    with pytest.raises(hosted.HostedAsrError) as err:
        hosted.transcribe(audio)
    assert "no segments" in str(err.value)
