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

echo "== pausing the burst schedule while we work"
ssh_ "sudo touch /opt/publikclip/no-burst && sudo shutdown -c 2>/dev/null; true"

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
ssh_ "sudo -u publikclip HOME=/opt/publikclip /opt/publikclip/.local/bin/uv \
  --directory /opt/publikclip/src/pipeline run publikclip publish '$JOB' \
  --dir '$REMOTE' --platforms '$PLATFORMS' --publish '$PUBLISH' \
  --visibility '$VISIBILITY' --clips '$CLIPS' --min-score '$MIN_SCORE'" || \
  echo "!! publishing reported a failure; the machine is still stopped below"

echo "== lifting the pause and stopping"
ssh_ "sudo rm -f /opt/publikclip/no-burst" || true
g compute instances stop "$NAME" --zone "$ZONE" >/dev/null
echo "== $NAME stopped"
