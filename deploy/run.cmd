@echo off
REM Run a publikclip command on the instance without opening a shell first.
REM
REM   run.cmd sources youtube "@Underscore_" --limit 3
REM   run.cmd auto --youtube "@channel" --limit 1 --llm nvidia
REM
REM The instance carries a /usr/local/bin/publikclip launcher that knows about
REM the service user, its HOME and its uv install — so nothing has to be
REM quoted twice on the way through cmd, gcloud and plink. Getting that
REM nesting right was not worth the fight.

if "%PROJECT%"=="" set PROJECT=gen-lang-client-0653010260
if "%ZONE%"=="" set ZONE=europe-west9-a
if "%NAME%"=="" set NAME=publikclip

gcloud compute ssh %NAME% --project %PROJECT% --zone %ZONE% --tunnel-through-iap --command "publikclip %*"
