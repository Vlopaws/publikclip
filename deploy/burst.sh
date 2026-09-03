#!/usr/bin/env bash
# One burst: wake, take what is new, publish it, shut down.
#
# The machine is billed by the second it is running, on a paid account whose
# remaining credit nobody can see. So the design goal is not "do the work" —
# it is "be off again", and everything below serves that.
#
# THE MACHINE MUST POWER OFF. Not "should". A burst that finishes cleanly
# powers off at the end; one that fails powers off in the trap; one that
# hangs is killed by systemd's RuntimeMaxSec and powers off in the trap
# anyway; and if all three of those fail, install-burst.sh has already armed
# a `shutdown` at the deadline from the boot itself. Four independent things
# have to go wrong before this bills a full day.
#
# Maintenance: `touch /opt/publikclip/no-burst` and the service does nothing
# and shuts nothing down, which is what makes the box safe to SSH into and
# work on. Remember to remove it.

set -uo pipefail   # NOT -e: a failing stage must still reach the shutdown

APP=/opt/publikclip
LOG=$APP/burst.log
PAUSE=$APP/no-burst
ROSTER=${ROSTER:-$APP/src/rosters/zevent.txt}
LIMIT=${LIMIT:-5}
PLATFORMS=${PLATFORMS:-tiktok,instagram}
PUBLISH=${PUBLISH:-dry-run}
VISIBILITY=${VISIBILITY:-private}

# The composite floor, lower here than the pipeline default of 40, and this
# needs saying plainly because lowering a threshold to make things pass is
# usually the wrong move.
#
# Measured, two populations that the rubric scores completely differently:
#
#     long-form talk (Underscore_, Thinkerview)   23 - 53
#     Twitch clips                                23 - 33   (4 clips)
#
# The rubric reads a transcript. A gaming clip's payload is visual — the
# kill, the reaction — and its transcript is "on s'arrpond". It is not that
# these clips are bad; it is that the instrument cannot see what makes them
# good, and one absolute floor across two populations it measures unequally
# rejects the whole of the second.
#
# What does the selecting on this path is upstream and stronger: the roster
# ranks every channel's clips by view count and --limit takes the head.
# Those views are a crowd choosing a highlight before the pipeline ever
# looks. The floor here is the weaker, second filter — so it drops to the
# bottom of the Twitch population rather than the middle of the other one.
#
# Fitted to four clips. Revisit it from real outcomes once anything has
# actually been posted.
MIN_SCORE=${MIN_SCORE:-30}
# Keep at least this much of the burst budget in reserve for shutting down.
POWEROFF=${POWEROFF:-1}

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

finish() {
  local code=$?
  say "burst ended (exit $code)"
  if [ "$POWEROFF" = "1" ] && [ ! -e "$PAUSE" ]; then
    say "powering off"
    sync
    systemctl poweroff --no-block || poweroff || shutdown -h now
  else
    say "poweroff suppressed (POWEROFF=$POWEROFF, pause=$([ -e "$PAUSE" ] && echo yes || echo no))"
  fi
}
trap finish EXIT

if [ -e "$PAUSE" ]; then
  # Deliberately before anything else, and it must not power off: this flag
  # exists so a person can work on the machine without it vanishing.
  POWEROFF=0
  say "paused by $PAUSE — doing nothing"
  exit 0
fi

say "=== burst starting: roster=$ROSTER limit=$LIMIT publish=$PUBLISH min_score=$MIN_SCORE ==="

run() {
  sudo -u publikclip HOME=$APP PYTHONUNBUFFERED=1 \
    "$APP/.local/bin/uv" --directory "$APP/src/pipeline" run publikclip "$@"
}

# Pull the day's code first: a burst is also how a fix reaches production.
sudo -u publikclip git -C "$APP/src" pull --ff-only -q 2>&1 | tee -a "$LOG" || \
  say "git pull failed, running the version already here"
say "version: $(sudo -u publikclip git -C "$APP/src" log --oneline -1)"

if [ ! -s "$ROSTER" ]; then
  say "roster $ROSTER is missing or empty — nothing to do"
  exit 0
fi

say "--- autopilot ---"
run auto \
  --twitch-roster "$ROSTER" \
  --limit "$LIMIT" \
  --llm groq \
  --min-score "$MIN_SCORE" \
  --platforms "$PLATFORMS" \
  --publish "$PUBLISH" \
  --visibility "$VISIBILITY" \
  2>&1 | tee -a "$LOG"

# Reclaim the source video and analysis audio from finished jobs. Left
# alone, three bursts a day fill the disk in a few weeks, and a full disk
# fails as whatever stage happened to be writing.
say "--- prune ---"
run jobs prune --older-than 1 --apply 2>&1 | tail -5 | tee -a "$LOG"

say "disk: $(df -h / | awk 'NR==2 {printf "%s free of %s", $4, $2}')"
