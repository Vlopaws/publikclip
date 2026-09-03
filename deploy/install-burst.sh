#!/usr/bin/env bash
# Install the burst service on this machine. Run once, as root, on the VM.
#
# Two independent guarantees that the machine powers off, because the whole
# point is not paying for a box that is idle or wedged:
#
#   1. systemd RuntimeMaxSec kills the burst at the deadline, and burst.sh's
#      EXIT trap powers off on the way out — including when it is killed.
#   2. A `shutdown` armed from the boot itself, as a backstop for the case
#      where the service never starts or systemd itself is unhappy. It is
#      cancelled only by the pause flag.
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

cat > /etc/systemd/system/publikclip-burst.service <<UNIT
[Unit]
Description=publikclip: one burst, then power off
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/publikclip-burst
# The deadline is a cost control, not a performance target. On expiry
# systemd kills the process and burst.sh's EXIT trap still powers off.
RuntimeMaxSec=${BUDGET_SEC}
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
# If the pause flag is set, someone is working on the machine: leave it up.
ExecStart=/bin/bash -c '[ -e ${APP}/no-burst ] || /sbin/shutdown -h +${BACKSTOP_MIN} "publikclip backstop: nothing should keep this machine up"'
ExecStop=/sbin/shutdown -c

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable publikclip-burst.service publikclip-backstop.service

echo "installed."
echo "  burst budget : ${BUDGET_SEC}s"
echo "  backstop     : shutdown +${BACKSTOP_MIN}min from boot"
echo
echo "pause both (to work on the machine):  sudo touch ${APP}/no-burst && sudo shutdown -c"
echo "resume:                               sudo rm ${APP}/no-burst"
