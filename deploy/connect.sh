#!/usr/bin/env bash
# A shell on the instance, brokered by IAP so no SSH port faces the internet.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-europe-west9-a}"
NAME="${NAME:-publikclip}"

exec gcloud compute ssh "$NAME" \
  --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap "$@"
