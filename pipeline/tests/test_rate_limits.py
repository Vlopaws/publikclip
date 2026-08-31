"""Backing off when the server says "too fast".

A rate limit is the expected steady state, not an exception: the scoring
stage fires one call per candidate — about a hundred for a two-hour source —
against a budget measured per minute. The first version treated it like any
other error and gave up after sleeping 1 s then 2 s, against a server asking
for 4. An autopilot run died at the scoring stage having already paid for
the download, the transcription, diarization and event detection.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.scoring import llm


class Res:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_the_retry_after_header_wins():
    assert llm.retry_delay(Res({"retry-after": "4"}), 1) == pytest.approx(4.5)


def test_the_servers_wording_is_read_when_there_is_no_header():
    # Groq's own phrasing, verbatim from the failure that prompted this.
    detail = (
        "Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per "
        "minute (TPM): Limit 8000, Used 7253, Requested 1285. Please try "
        "again in 4.035s."
    )
    assert llm.retry_delay(Res(), 1, detail) == pytest.approx(4.535)


def test_milliseconds_are_understood_as_milliseconds():
    delay = llm.retry_delay(Res(), 1, "Please try again in 250ms.")
    assert delay == pytest.approx(0.75)


def test_it_always_waits_a_little_longer_than_asked():
    # Returning at the exact deadline just burns another attempt.
    assert llm.retry_delay(Res({"retry-after": "2"}), 0) > 2.0
    assert llm.retry_delay(Res(), 0, "try again in 1s") > 1.0


def test_it_falls_back_to_exponential_backoff_when_told_nothing():
    assert llm.retry_delay(Res(), 1) == pytest.approx(2.0)
    assert llm.retry_delay(Res(), 3) == pytest.approx(8.0)


def test_a_garbage_header_does_not_crash_the_retry():
    assert llm.retry_delay(Res({"retry-after": "soon"}), 2) == pytest.approx(4.0)


def test_no_wait_runs_away_unboundedly():
    # A server asking for an hour should surface as an error, not as a
    # process that looks hung.
    assert llm.retry_delay(Res({"retry-after": "3600"}), 1) == llm.RATE_LIMIT_MAX_WAIT
    assert llm.retry_delay(Res(), 40) == llm.RATE_LIMIT_MAX_WAIT
    assert llm.retry_delay(Res(), 1, "try again in 900s") == llm.RATE_LIMIT_MAX_WAIT


def test_rate_limits_get_more_attempts_than_hard_failures():
    # Waiting is always the right answer to "too fast" and never the right
    # answer to "wrong key", so the two budgets must not be the same one.
    assert llm.RATE_LIMIT_ATTEMPTS > 3
