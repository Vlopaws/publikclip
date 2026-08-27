#!/usr/bin/env bash
# Postiz on http://localhost:4007, tunnelled from the instance.
#
# The instance publishes nothing to the internet: Postiz holds the OAuth
# tokens for every connected social account, so it is reached through this
# tunnel rather than an open port. Leave this running while you use it.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-europe-west9-a}"
NAME="${NAME:-publikclip}"
PORT="${PORT:-4007}"

echo "Postiz → http://localhost:${PORT}   (ctrl-c to close)"
exec gcloud compute start-iap-tunnel "$NAME" 4007 \
  --local-host-port="localhost:${PORT}" \
  --project "$PROJECT" --zone "$ZONE"
