# Running publikclip on Google Cloud

One Compute Engine instance carries the whole thing: the pipeline, the Postiz
publishing stack, and the scheduled runs. Scoring and transcription are hosted
APIs, so nothing here needs a GPU.

## What it costs

`e2-standard-4` (4 vCPU, 16 GB) in `europe-west9` runs continuously. The disk
is 150 GB because source videos and rendered clips are not small; model weights
are only ~2 GB of it.

Nothing else is billable unless you add it. Scoring runs on NVIDIA Build's free
tier and transcription on Groq, both outside GCP.

If the box is idle most of the day, stop it between runs — a stopped instance
bills only for its disk:

```sh
gcloud compute instances stop publikclip --zone europe-west9-a
```

## Nothing is exposed

The instance takes no inbound connections. SSH is brokered by Identity-Aware
Proxy, and Postiz is reached through a local tunnel rather than a public port.

That matters more than usual here: Postiz holds the OAuth tokens for every
connected social account. A self-hosted instance on an open port, with a
default install, is how those get taken.

## Sequence

You authenticate; these scripts never ask for credentials.

```sh
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
export PROJECT=YOUR_PROJECT_ID
```

Then, from this directory:

```sh
./provision.sh          # creates the instance, enables IAP
```

First boot installs Docker, uv, ffmpeg, the pipeline and Postiz. It takes a few
minutes. Watch it:

```sh
./connect.sh --command 'sudo tail -f /var/log/publikclip-setup.log'
```

## Using it

```sh
./connect.sh            # a shell on the box
./tunnel.sh             # Postiz at http://localhost:4007
```

Secrets live in `~/.publikclip/secrets.json` on the instance, the same as
locally. Put the NVIDIA, Groq and Postiz keys there — they are not copied from
your laptop, deliberately.

## Scheduled runs

`publikclip auto` is a single command, so a systemd timer or cron is enough;
the job queue is what stops the same video being processed twice.

```sh
cd /opt/publikclip/src/pipeline
uv run publikclip auto --youtube "@channel" --limit 1 --llm nvidia \
  --publish postiz --platforms tiktok
```

Start with the default `--publish dry-run` until the output convinces you.
