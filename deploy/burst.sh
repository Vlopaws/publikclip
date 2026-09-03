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

say "=== burst starting: roster=$ROSTER limit=$LIMIT publish=$PUBLISH ==="

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
  --platforms "$PLATFORMS" \
  --publish "$PUBLISH" \
  --visibility "$VISIBILITY" \
  2>&1 | tee -a "$LOG"

# Reclaim the source video and analysis audio from finished jobs. Left
# alone, three bursts a day fill the disk in a few weeks, and a full disk
# fails as whatever stage happened to be writing.
say "--- prune ---"
run jobs prune --older-than 1 --apply 2>&1 | tail -5 | tee -a "$LOG"

say "disk: $(df -h / | awk 'NR==2 {print $4\" free of \"$2}')"
