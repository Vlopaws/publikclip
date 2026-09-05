#!/usr/bin/env bash
# Power off a machine that has no reason to be on.
#
# The lease was the missing half of a promise. burst.sh reads it once, at
# the end of a burst, so a lease that expires triggers nothing: the machine
# that had been held stays held until the next boot, which is to say until
# somebody notices. Measured, that was six hours.
#
# This is the half that acts. Every few minutes it asks one question — is
# there a reason for this machine to be running — and if there is not, it
# powers off. It is the only guarantee here that does not depend on a burst
# having started, a trap having fired, or systemd honouring a directive.
set -uo pipefail

APP=/opt/publikclip
LEASE=$APP/keep-up-until
LOG=$APP/reaper.log
# A boot needs room to start its burst before anything judges it idle.
GRACE_SEC=${GRACE_SEC:-900}

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

# A burst in progress is a reason.
if systemctl is-active --quiet publikclip-burst; then
  exit 0
fi

# So is a boot that is still getting started.
uptime_sec=$(cut -d. -f1 /proc/uptime 2>/dev/null || echo 99999)
if [ "$uptime_sec" -lt "$GRACE_SEC" ]; then
  exit 0
fi

# And so is a lease, while it lasts. Anything malformed, empty, expired or
# missing is not a reason: the default has to be off, because the failure
# this exists to prevent is the machine staying on.
if [ -r "$LEASE" ]; then
  until=$(cat "$LEASE" 2>/dev/null || echo "")
  case "$until" in
    ""|*[!0-9]*) ;;
    *)
      if [ "$(date +%s)" -lt "$until" ]; then
        exit 0
      fi
      say "lease expired at $(date -Is -d "@$until")"
      ;;
  esac
fi

say "no burst, no lease, up ${uptime_sec}s — powering off"
sync
systemctl poweroff --no-block || poweroff || shutdown -h now
