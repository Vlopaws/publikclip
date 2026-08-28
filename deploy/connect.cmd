@echo off
REM A shell on the instance, brokered by IAP so no SSH port faces the internet.
REM
REM The Windows twin of connect.sh: cmd.exe understands neither `VAR=x cmd`
REM nor `./script.sh`, so the operator on Windows needs its own entry point.

if "%PROJECT%"=="" set PROJECT=gen-lang-client-0653010260
if "%ZONE%"=="" set ZONE=europe-west9-a
if "%NAME%"=="" set NAME=publikclip

gcloud compute ssh %NAME% --project %PROJECT% --zone %ZONE% --tunnel-through-iap %*
