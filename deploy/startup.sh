#!/usr/bin/env bash
# Runs once, on the instance's first boot, as root.
#
# Installs what the pipeline needs to run unattended: Docker for the Postiz
# stack, uv for the Python environment, and a system ffmpeg so neither our
# code nor whisperx has to hunt for one.
#
# No local model server: scoring runs on NVIDIA Build and transcription on
# Groq, both hosted. Running an 8B model here would cost GPU hours to get a
# worse judge than the free API already provides.
#
# Logs land in /var/log/publikclip-setup.log — read that before believing
# anything went wrong.
set -euo pipefail
exec > >(tee -a /var/log/publikclip-setup.log) 2>&1
echo "=== publikclip setup $(date -Is) ==="

meta() {
  curl -fsH "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}

REPO_URL="$(meta publikclip-repo)"
REPO_REF="$(meta publikclip-ref)"
APP_USER="publikclip"
APP_HOME="/opt/publikclip"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# ffmpeg from the distro: the pipeline can fetch its own, but having one on
# PATH means whisperx and every other library that shells out finds it too.
apt-get install -y --no-install-recommends \
  ca-certificates curl git ffmpeg python3 build-essential

# --- Docker ---------------------------------------------------------------
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# --- the service account it runs as ---------------------------------------
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "$APP_HOME" --shell /bin/bash "$APP_USER"
usermod -aG docker "$APP_USER"

# --- the project ----------------------------------------------------------
if [ -n "$REPO_URL" ]; then
  sudo -u "$APP_USER" git clone --branch "${REPO_REF:-main}" "$REPO_URL" "$APP_HOME/src" || \
    sudo -u "$APP_USER" git clone "$REPO_URL" "$APP_HOME/src"
fi

# uv resolves Python 3.12 and the pipeline's dependencies by itself.
sudo -u "$APP_USER" bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
if [ -d "$APP_HOME/src/pipeline" ]; then
  sudo -u "$APP_USER" bash -lc "cd $APP_HOME/src/pipeline && \$HOME/.local/bin/uv sync"
fi

# --- Postiz ---------------------------------------------------------------
# Cloned rather than vendored: its compose file is maintained upstream and
# pinning a snapshot here would rot.
sudo -u "$APP_USER" git clone --depth 1 \
  https://github.com/gitroomhq/postiz-docker-compose.git "$APP_HOME/postiz" || true

if [ -f "$APP_HOME/postiz/docker-compose.yaml" ]; then
  # The shipped compose carries a placeholder JWT secret in plain text.
  SECRET="$(head -c 48 /dev/urandom | base64 | tr -d '\n=+/')"
  sed -i "s|random string that is unique to every install - just type random characters here!|${SECRET}|" \
    "$APP_HOME/postiz/docker-compose.yaml"
  sudo -u "$APP_USER" bash -lc "cd $APP_HOME/postiz && docker compose up -d"
fi

echo "=== setup finished $(date -Is) ==="
