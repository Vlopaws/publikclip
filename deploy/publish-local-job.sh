#!/usr/bin/env bash
# Publish a job that was rendered on the operator's machine.
#
# YouTube refuses downloads from the VM's datacentre address, so a YouTube
# source is cut locally. The publishing key lives on the VM and stays there.
# This carries the small end — the finished clips and the two checkpoints
# that describe them — up to the box that can post, posts, and puts the box
# back to sleep.
#
# What travels: clips/*.mp4, render.json, score.json. Not the 600 MB source,
# not the analysis audio; neither is needed to publish and both would take
# ten minutes to upload.
#
# The pause flag goes on before anything else. Without it the boot-time
# backstop would power the machine off a hundred minutes later, possibly
# mid-upload, and the burst service would start a roster run underneath this
# one.
set -euo pipefail

JOB=${1:?usage: publish-local-job.sh JOB_ID [platforms]}
PLATFORMS=${2:-tiktok}
VISIBILITY=${VISIBILITY:-private}
CLIPS=${CLIPS:-3}
MIN_SCORE=${MIN_SCORE:-40}
PUBLISH=${PUBLISH:-dry-run}
# Exact clip numbers, when the choice has already been made by someone who
# watched them. Empty means select by score.
ONLY=${ONLY:-}
# Minutes between clips, and before the first one. Zernio holds a scheduled
# post itself, so the machine only has to stay up long enough to upload the
# files and create the posts -- not until they go out.
STAGGER=${STAGGER:-0}
DELAY=${DELAY:-0}
# How long the machine is held up for this publish. Long enough to upload
# and post, short enough that a crash here costs minutes, not a night.
LEASE_MIN=${LEASE_MIN:-45}
# The box's own remote name, which is not this machine's. Hard-coding "fork"
# here failed silently: git printed its complaint, the script carried on
# with the old checkout, and the flags it then passed did not exist.
REMOTE=${REMOTE:-origin}
BRANCH=${BRANCH:-hardening-and-llm-backends}

PROJECT=${PROJECT:-gen-lang-client-0653010260}
ZONE=${ZONE:-europe-west9-a}
NAME=${NAME:-publikclip}
LOCAL_JOBS=${LOCAL_JOBS:-$HOME/.publikclip/jobs}
REMOTE=/tmp/incoming/$JOB

g() { gcloud "$@" --project "$PROJECT"; }
ssh_() { g compute ssh "$NAME" --zone "$ZONE" --tunnel-through-iap --command "$1"; }

echo "== starting $NAME"
g compute instances start "$NAME" --zone "$ZONE" >/dev/null
# Give sshd a moment; the start call returns before the box answers.
until ssh_ "true" 2>/dev/null; do sleep 10; done

# A lease, not a flag. The flag this used to set stopped the burst AND every
# power-off path, and one left behind kept the machine up for six hours. A
# lease expires whether or not this script reaches its last line -- which
# matters most exactly when it does not.
echo "== taking a ${LEASE_MIN}min lease on the machine"
ssh_ "sudo sh -c 'date -d \"+${LEASE_MIN} min\" +%s > /opt/publikclip/keep-up-until'   && sudo touch /opt/publikclip/no-burst && sudo shutdown -c 2>/dev/null; true"

# The flags this script passes have to exist on the box that runs them. A
# publish that fails on an unknown argument has already started the machine,
# uploaded the clips and woken the operator.
echo "== updating the checkout"
ssh_ "sudo git config --global --add safe.directory /opt/publikclip/src;   sudo -u publikclip git -C /opt/publikclip/src fetch --quiet $REMOTE &&   sudo -u publikclip git -C /opt/publikclip/src checkout --quiet -B $BRANCH   $REMOTE/$BRANCH && sudo git -C /opt/publikclip/src log --oneline -1"

echo "== shipping the job"
ssh_ "mkdir -p $REMOTE/clips"
g compute scp --zone "$ZONE" --tunnel-through-iap \
  "$LOCAL_JOBS/$JOB/render.json" "$LOCAL_JOBS/$JOB/score.json" "$NAME:$REMOTE/"
g compute scp --zone "$ZONE" --tunnel-through-iap --recurse \
  "$LOCAL_JOBS/$JOB/clips" "$NAME:$REMOTE/"

# render.json records absolute paths from the machine that rendered it.
echo "== repointing the clip paths"
ssh_ "python3 - <<'PY'
import json, pathlib, re
p = pathlib.Path('$REMOTE/render.json')
d = json.loads(p.read_text(encoding='utf-8'))
for o in d['data']['outputs']:
    o['path'] = '$REMOTE/clips/' + re.split(r'[\\\\\\\\/]', o['path'])[-1]
    if o.get('ass'):
        o['ass'] = '$REMOTE/clips/' + re.split(r'[\\\\\\\\/]', o['ass'])[-1]
p.write_text(json.dumps(d), encoding='utf-8')
print('repointed', len(d['data']['outputs']), 'clips')
PY"

echo "== publishing ($PUBLISH, $VISIBILITY, $PLATFORMS)"
echo "   clips=${ONLY:-top $CLIPS above $MIN_SCORE}; first in ${DELAY}min, then every ${STAGGER}min"
ssh_ "sudo -u publikclip HOME=/opt/publikclip /opt/publikclip/.local/bin/uv \
  --directory /opt/publikclip/src/pipeline run publikclip publish '$JOB' \
  --dir '$REMOTE' --platforms '$PLATFORMS' --publish '$PUBLISH' \
  --visibility '$VISIBILITY' --clips '$CLIPS' --min-score '$MIN_SCORE' \
  --stagger '$STAGGER' --delay '$DELAY' \
  ${ONLY:+--only '$ONLY'}" || \
  echo "!! publishing reported a failure; the machine is still stopped below"

echo "== releasing the lease and stopping"
ssh_ "sudo rm -f /opt/publikclip/keep-up-until /opt/publikclip/no-burst" || true
g compute instances stop "$NAME" --zone "$ZONE" >/dev/null
echo "== $NAME stopped"
