"""Clips that continue each other, labelled as such.

Two clips cut back to back from the same moment are one thing in two
pieces, and a viewer who sees the second first has no way to know that.
Saying "1/2" and "2/2" on them costs six characters and turns two
disconnected forty-second videos into a reason to look for the other one.

Adjacency is the whole signal, and it is unambiguous: the second starts
where the first ends. Nothing here guesses at meaning — two clips about the
same subject an hour apart are not parts, and clips that touch are, whether
the pipeline chose the boundary or the operator dictated it.
"""

from __future__ import annotations

# How close two clips must be to count as continuing each other. Sentence
# snapping and transition nudging move an edge by a second or two, so exact
# equality would almost never hold; a full second of daylight between them
# means they are not the same moment.
ADJACENT_GAP_SEC = 2.5

# Below this a "part" label is noise: a run of eight clips is not a serial,
# it is the whole video, and numbering it that way says nothing.
MAX_PART_RUN = 4


def group(bounds: list[tuple[float, float]]) -> dict[int, tuple[int, int]]:
    """index -> (part number, total), for clips that continue each other.

    Takes (start, end) per clip in any order and returns only the members
    of a run of two or more. A clip that stands alone is absent from the
    result rather than reported as 1/1, which would be a label that says
    nothing about anything.
    """
    order = sorted(range(len(bounds)), key=lambda i: bounds[i][0])
    runs: list[list[int]] = []
    current: list[int] = []

    for idx in order:
        if not current:
            current = [idx]
            continue
        previous_end = bounds[current[-1]][1]
        if 0 <= bounds[idx][0] - previous_end <= ADJACENT_GAP_SEC:
            current.append(idx)
        else:
            runs.append(current)
            current = [idx]
    if current:
        runs.append(current)

    labels: dict[int, tuple[int, int]] = {}
    for run in runs:
        if not (2 <= len(run) <= MAX_PART_RUN):
            continue
        for position, idx in enumerate(run, start=1):
            labels[idx] = (position, len(run))
    return labels


def label(title: str, part: tuple[int, int] | None) -> str:
    """Append the part marker to a title that has one.

    Appended rather than passed to the model: which clip is second is a
    fact about the cut, known here and not there, and asking the model to
    say it invites it to say something else.
    """
    if not title or not part:
        return title
    return f"{title} ({part[0]}/{part[1]})"
