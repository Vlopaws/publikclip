#!/usr/bin/env bash
# Install the burst service on this machine. Run once, as root, on the VM.
#
# Two guarantees that the machine powers off, because the whole point is not
# paying for a box that is idle or wedged:
#
#   1. systemd TimeoutStartSec kills the burst at the deadline, and
#      burst.sh's EXIT trap powers off on the way out — including when it is
#      killed. It was RuntimeMaxSec, which systemd ignores for Type=oneshot
#      and said so in the journal on every boot: "RuntimeMaxSec= has no
#      effect in combination with Type=oneshot. Ignoring." A guarantee that
#      announces its own absence once a day is worse than no guarantee,
#      because it was counted.
#   2. A `shutdown` armed from the boot itself, as a backstop for the case
#      where the service never starts or systemd itself is unhappy.
#
# Both are now armed unconditionally. Previously both, plus burst.sh's own
# power-off, were gated on one file — so they were not independent at all,
# and one stray flag kept the machine up for six hours. Only a lease with a
# deadline holds the machine now, and it expires by itself.
#
# Nothing here starts the machine. That cannot come from inside a box that
# is off — see the instance schedule in README.md.
set -euo pipefail

APP=/opt/publikclip
# Long enough for five clips end to end with room to spare; short enough
# that a wedged burst costs an hour and a half, not a day.
BUDGET_SEC=${BUDGET_SEC:-5400}
BACKSTOP_MIN=${BACKSTOP_MIN:-100}

install -o root -g root -m 755 "$APP/src/deploy/burst.sh" /usr/local/bin/publikclip-burst

# Settings live outside the unit so they can be changed without editing
# systemd or the repo — and so a change survives the next reinstall.
if [ ! -f "$APP/burst.env" ]; then
  cat > "$APP/burst.env" <<'ENV'
# publikclip burst settings. Edit, then: sudo systemctl daemon-reload
#
# PUBLISH     dry-run | zernio      (dry-run posts nothing at all)
# VISIBILITY  private | unlisted | public
# LIMIT       how many clips one burst processes; each costs ~5 min
# MIN_SCORE   composite floor; see the note in burst.sh before raising it
PUBLISH=zernio
VISIBILITY=private
LIMIT=5
PLATFORMS=tiktok,instagram
MIN_SCORE=30
ROSTER=/opt/publikclip/src/rosters/zevent.txt
ENV
  chmod 644 "$APP/burst.env"
fi

cat > /etc/systemd/system/publikclip-burst.service <<UNIT
[Unit]
Description=publikclip: one burst, then power off
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
EnvironmentFile=-${APP}/burst.env
ExecStart=/usr/local/bin/publikclip-burst
# The deadline is a cost control, not a performance target. On expiry
# systemd kills the process and burst.sh's EXIT trap still powers off.
# TimeoutStartSec, not RuntimeMaxSec: for Type=oneshot the whole run counts
# as start-up, and RuntimeMaxSec is ignored outright.
TimeoutStartSec=${BUDGET_SEC}
TimeoutStopSec=120
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/publikclip-backstop.service <<UNIT
[Unit]
Description=publikclip: arm a shutdown from boot, in case the burst never runs
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
# Armed on every boot, no exceptions. Whoever wants the machine to stay up
# takes a lease and cancels this themselves; a lease that runs out lets the
# next boot arm it again. The old version skipped arming whenever a pause
# flag existed, which made the backstop depend on exactly the thing it was
# supposed to back up.
ExecStart=/bin/bash -c '/sbin/shutdown -h +${BACKSTOP_MIN} "publikclip backstop: nothing should keep this machine up"'
ExecStop=/sbin/shutdown -c

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable publikclip-burst.service publikclip-backstop.service

echo "installed."
echo "  settings     : ${APP}/burst.env"
echo "  burst budget : ${BUDGET_SEC}s"
echo "  backstop     : shutdown +${BACKSTOP_MIN}min from boot"
echo
echo "do no work this boot :  sudo touch ${APP}/no-burst      (still powers off)"
echo "keep it up 60 min    :  sudo sh -c 'date -d \"+60 min\" +%s > ${APP}/keep-up-until' && sudo shutdown -c"
echo "release early        :  sudo rm -f ${APP}/keep-up-until"
echo
echo "A lease expires on its own. That is the point: forgetting a flag is free,"
echo "forgetting a running VM is not."
