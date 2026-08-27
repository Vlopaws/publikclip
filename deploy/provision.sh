#!/usr/bin/env bash
# Create the Compute Engine instance publikclip runs on.
#
# Run this from a machine where `gcloud auth login` has already been done —
# it never asks for credentials itself.
#
# Nothing is exposed to the internet. Postiz and any other UI are reached
# through an IAP tunnel (see connect.sh), because a freshly installed
# self-hosted app on a public port is a bad first day.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your GCP project id}"
ZONE="${ZONE:-europe-west9-a}"          # Paris: closest to a French operator
NAME="${NAME:-publikclip}"

# 4 vCPU / 16 GB carries the pipeline (PANNs, scene detection, rendering) and
# the Postiz containers. Scoring and transcription are hosted APIs, so there
# is nothing here a GPU would accelerate.
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"

# Model weights are ~2 GB, but source videos and rendered clips are not small.
DISK_GB="${DISK_GB:-150}"

REPO_URL="${REPO_URL:-https://github.com/Blueturboguy07/publikclip.git}"
REPO_REF="${REPO_REF:-hardening-and-llm-backends}"

echo "project      : $PROJECT"
echo "zone         : $ZONE"
echo "machine      : $MACHINE_TYPE   disk ${DISK_GB}GB"
echo

gcloud services enable compute.googleapis.com iap.googleapis.com \
  --project "$PROJECT"

# Deliberately no external HTTP/HTTPS tags: the instance keeps its outbound
# access (it downloads videos and models) and accepts nothing inbound except
# IAP-brokered SSH.
gcloud compute instances create "$NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --machine-type "$MACHINE_TYPE" \
  --image-family ubuntu-2404-lts-amd64 \
  --image-project ubuntu-os-cloud \
  --boot-disk-size "${DISK_GB}GB" \
  --boot-disk-type pd-balanced \
  --metadata-from-file startup-script=./startup.sh \
  --metadata "publikclip-repo=${REPO_URL},publikclip-ref=${REPO_REF}" \
  --scopes cloud-platform \
  --labels app=publikclip

# IAP brokers SSH without the instance holding a public SSH port open.
if ! gcloud compute firewall-rules describe allow-iap-ssh --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute firewall-rules create allow-iap-ssh \
    --project "$PROJECT" \
    --direction INGRESS \
    --action allow \
    --rules tcp:22 \
    --source-ranges 35.235.240.0/20 \
    --description "SSH via Identity-Aware Proxy only"
fi

echo
echo "Instance created. First boot installs Docker, uv, ffmpeg and the"
echo "pipeline; give it a few minutes, then:"
echo
echo "  ./connect.sh          # shell on the box"
echo "  ./tunnel.sh           # Postiz at http://localhost:4007 through IAP"
