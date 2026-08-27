"""Unattended clipping: discover, process, select, publish.

Every stage of it already existed as a manual step; this is the wiring that
lets them run without a person in the loop, plus the two things that only
matter once nobody is watching — a ledger so the same clip is never posted
twice, and a dry run that shows exactly what would go out before anything
does.

Publishing defaults to posting nothing. That is not a placeholder: an
automated poster is the one part of this pipeline that cannot be undone, so
it should be switched on deliberately, after a rehearsal.
"""

from . import postiz, publish, runner, select
from .publish import PublishError, PublishResult, make_publisher
from .runner import RunReport, run
from .select import SelectedClip

__all__ = [
    "PublishError",
    "PublishResult",
    "RunReport",
    "SelectedClip",
    "make_publisher",
    "postiz",
    "publish",
    "run",
    "runner",
    "select",
]
