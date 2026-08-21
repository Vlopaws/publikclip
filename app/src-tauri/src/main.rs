// publikclip desktop shell. The pipeline is a Python sidecar speaking JSONL
// on stdout (`publikclip --jsonl ...`); this shell spawns it, forwards every
// event to the frontend, and exposes small filesystem/settings commands.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};

fn home_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("PUBLIKCLIP_HOME") {
        return PathBuf::from(custom);
    }
    dirs_home().join(".publikclip")
}

fn dirs_home() -> PathBuf {
    // HOME on Unix; Windows services and some launch paths only set USERPROFILE.
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Absolute path to the OS-owned curl, or None when it is not there.
///
/// `Command::new("curl")` resolves through the application's own directory
/// before PATH on Windows, so a `curl.exe` dropped beside the installed app
/// (or planted earlier on PATH) would run instead of the real one. Anything
/// the app launches on its own initiative takes the system copy or nothing.
fn system_curl() -> Option<PathBuf> {
    let candidate = if cfg!(windows) {
        PathBuf::from(std::env::var("SystemRoot").unwrap_or_else(|_| String::from("C:/Windows")))
            .join("System32")
            .join("curl.exe")
    } else {
        PathBuf::from("/usr/bin/curl")
    };
    candidate.exists().then_some(candidate)
}

/// Subcommands the UI may reach through the generic `edit_tool` / `ig_tool`
/// bridges, which forward a caller-supplied argv to the pipeline CLI.
///
/// No shell is involved, so this is not about shell injection — it is about
/// not handing the whole CLI surface to whatever executes in the webview,
/// and about new subcommands being opt-in rather than exposed the moment
/// they are added. `ig connect` is excluded deliberately: it takes a Meta
/// app secret and has its own command with its own handling.
const EDIT_SUBCOMMANDS: &[&str] = &["context", "suggest-visuals"];
const IG_SUBCOMMANDS: &[&str] =
    &["sync", "overview", "media", "link", "unlink", "reject", "pull", "report"];

fn checked_subcommand(args: &[String], allowed: &[&str], tool: &str) -> Result<(), String> {
    match args.first() {
        Some(sub) if allowed.contains(&sub.as_str()) => Ok(()),
        Some(sub) => Err(format!("{tool}: subcommand `{sub}` is not allowed")),
        None => Err(format!("{tool}: no subcommand given")),
    }
}

/// Command that never flashes a console window on Windows (CREATE_NO_WINDOW).
/// Every pipeline/tool spawn goes through this — a GUI app popping cmd.exe
/// windows for each sidecar call reads as malware to most people.
fn quiet_command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Where the Python pipeline lives and how to invoke it.
/// Dev builds call `uv run` against the repo's pipeline/ directory. Packaged
/// builds invoke the bundled python env (M6); resolution stays in one place.
fn pipeline_invocation() -> (String, Vec<String>) {
    if cfg!(debug_assertions) {
        let pipeline_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../pipeline")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("../pipeline"));
        (
            "uv".to_string(),
            vec![
                "--directory".to_string(),
                pipeline_dir.to_string_lossy().to_string(),
                "run".to_string(),
                "publikclip".to_string(),
            ],
        )
    } else {
        // Packaged: bundled uv + pipeline source under the platform's
        // resource layout — macOS keeps them in the .app's Resources dir,
        // Windows (NSIS) lands them in resources\ next to the exe. The venv
        // bootstraps into PUBLIKCLIP_HOME on first run (uv handles Python
        // 3.12 download + deps; the onboarding screen owns expectations).
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let resources = if cfg!(target_os = "macos") {
            exe_dir.join("../Resources/resources")
        } else {
            exe_dir.join("resources")
        };
        let uv = if cfg!(target_os = "windows") { "bin/uv.exe" } else { "bin/uv" };
        (
            resources.join(uv).to_string_lossy().to_string(),
            vec![
                "--directory".to_string(),
                resources.join("pipeline").to_string_lossy().to_string(),
                "run".to_string(),
                "publikclip".to_string(),
            ],
        )
    }
}

#[tauri::command]
fn run_job(app: AppHandle, source: String, llm: Option<String>, captions: Option<String>) -> Result<(), String> {
    let (program, base_args) = pipeline_invocation();
    std::thread::spawn(move || {
        let mut args = base_args.clone();
        args.push("--jsonl".to_string());
        args.push("run".to_string());
        args.push(source);
        if let Some(mode) = llm {
            args.push("--llm".to_string());
            args.push(mode);
        }
        if let Some(preset) = captions {
            args.push("--captions".to_string());
            args.push(preset);
        }
        stream_pipeline(&app, &program, &args);
    });
    Ok(())
}

#[tauri::command]
fn resume_job(
    app: AppHandle,
    job_id: String,
    llm: Option<String>,
    captions: Option<String>,
    camera: Option<String>,
) -> Result<(), String> {
    let (program, base_args) = pipeline_invocation();
    std::thread::spawn(move || {
        let mut args = base_args.clone();
        args.push("--jsonl".to_string());
        args.push("resume".to_string());
        args.push(job_id);
        if let Some(mode) = llm {
            args.push("--llm".to_string());
            args.push(mode);
        }
        if let Some(preset) = captions {
            args.push("--captions".to_string());
            args.push(preset);
        }
        if let Some(cam) = camera {
            args.push("--camera".to_string());
            args.push(cam);
        }
        stream_pipeline(&app, &program, &args);
    });
    Ok(())
}

fn stream_pipeline(app: &AppHandle, program: &str, args: &[String]) {
    let child = quiet_command(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(err) => {
            let _ = app.emit(
                "pipeline-event",
                json!({"event": "result", "ok": false, "error": format!("could not start pipeline: {err}")}),
            );
            return;
        }
    };
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                let _ = app.emit("pipeline-event", value);
            }
        }
    }
    if let Ok(status) = child.wait() {
        if !status.success() {
            let _ = app.emit("pipeline-event", json!({"event": "exited", "code": status.code()}));
        }
    }
}

/// Everything the review UI needs for one job, read straight off the job
/// dir's checkpoint files (artifacts are the truth).
#[tauri::command]
fn job_results(job_id: String) -> Result<Value, String> {
    let dir = home_dir().join("jobs").join(&job_id);
    if !dir.exists() {
        return Err(format!("no job dir for {job_id}"));
    }
    let read_stage = |name: &str| -> Value {
        fs::read_to_string(dir.join(format!("{name}.json")))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.get("data").cloned())
            .unwrap_or(Value::Null)
    };
    Ok(json!({
        "job_id": job_id,
        "dir": dir.to_string_lossy(),
        "ingest": read_stage("ingest"),
        "score": read_stage("score"),
        "camera": read_stage("camera"),
        "render": read_stage("render"),
        "events": read_stage("events"),
        "candidates": read_stage("candidates"),
    }))
}

#[tauri::command]
fn list_job_dirs() -> Result<Vec<Value>, String> {
    let jobs_dir = home_dir().join("jobs");
    let mut out = vec![];
    if let Ok(entries) = fs::read_dir(&jobs_dir) {
        for entry in entries.flatten() {
            let id = entry.file_name().to_string_lossy().to_string();
            let dir = entry.path();
            let has_render = dir.join("render.json").exists();
            let has_ingest = dir.join("ingest.json").exists();
            let title = fs::read_to_string(dir.join("ingest.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .and_then(|v| v["data"]["title"].as_str().map(String::from));
            out.push(json!({
                "id": id, "title": title,
                "ingested": has_ingest, "rendered": has_render,
            }));
        }
    }
    out.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
    Ok(out)
}

/// Persist one field into PUBLIKCLIP_HOME/secrets.json.
///
/// Both API keys go through here so the owner-only tightening cannot be
/// applied to one and forgotten on the other — which is exactly what had
/// happened: the Gemini key was chmod'ed 0600 and the Pexels key, written by
/// a near-copy of this function, was left at the process umask.
///
/// Windows deliberately keeps the inherited profile ACL (SYSTEM,
/// Administrators, the owning user — no other standard user). Rewriting it
/// with icacls would buy nothing an administrator could not undo anyway,
/// while risking locking the owner out of their own key.
fn write_secret(field: &str, value: &str) -> Result<bool, String> {
    let home = home_dir();
    fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    let path = home.join("secrets.json");
    let mut current: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    current[field] = json!(value.trim());
    fs::write(&path, serde_json::to_string_pretty(&current).unwrap()).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    }
    Ok(true)
}

#[tauri::command]
fn save_gemini_key(key: String) -> Result<bool, String> {
    write_secret("gemini_api_key", &key)
}

/// Secret fields the UI is allowed to write.
///
/// An allowlist for a sharper reason than the tool bridges have one:
/// secrets.json also carries `llm_base_url`, which the pipeline POSTs every
/// prompt to. A webview able to name arbitrary fields could point the whole
/// scoring stage — transcripts included — at a host of its choosing. Custom
/// endpoints stay env-var or hand-edited on purpose.
const WRITABLE_SECRETS: &[&str] = &[
    "gemini_api_key",
    "pexels_api_key",
    "nvidia_api_key",
    "openrouter_api_key",
    "openai_api_key",
    "groq_api_key",
];

#[tauri::command]
fn save_provider_key(field: String, key: String) -> Result<bool, String> {
    if !WRITABLE_SECRETS.contains(&field.as_str()) {
        return Err(format!("`{field}` is not a UI-writable secret"));
    }
    write_secret(&field, &key)
}

#[tauri::command]
fn get_setup_state() -> Result<Value, String> {
    let stored: Value = fs::read_to_string(home_dir().join("secrets.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    let has = |field: &str| {
        stored[field].as_str().map(|k| !k.trim().is_empty()).unwrap_or(false)
    };
    // Which brains are actually usable, so the picker can say so rather than
    // letting the user start an hour-long job that dies on the first call.
    let mut keys = serde_json::Map::new();
    for field in WRITABLE_SECRETS {
        keys.insert((*field).to_string(), json!(has(field)));
    }
    let onboarded = home_dir().join("onboarded").exists();
    Ok(json!({
        "has_gemini_key": has("gemini_api_key"),
        "onboarded": onboarded,
        "keys": Value::Object(keys),
    }))
}

#[tauri::command]
fn mark_onboarded() -> Result<(), String> {
    let home = home_dir();
    fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    fs::write(home.join("onboarded"), "1").map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_ollama() -> Result<Value, String> {
    let Some(curl) = system_curl() else {
        return Ok(json!({"running": false, "models": []}));
    };
    let out = quiet_command(&curl.to_string_lossy())
        .args(["-s", "-m", "3", "http://localhost:11434/api/tags"])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Ok(json!({"running": false, "models": []}));
    }
    let parsed: Value = serde_json::from_slice(&out.stdout).unwrap_or(json!({}));
    let models: Vec<String> = parsed["models"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect())
        .unwrap_or_default();
    Ok(json!({"running": true, "models": models}))
}

/// Sync pipeline call that returns one JSON blob (edit context, visual
/// suggestions). Long-running render-clip goes through run_edit_render
/// instead so progress streams.
#[tauri::command]
async fn edit_tool(args: Vec<String>) -> Result<Value, String> {
    checked_subcommand(&args, EDIT_SUBCOMMANDS, "edit")?;
    let (program, base_args) = pipeline_invocation();
    let mut full = base_args;
    full.push("edit".to_string());
    full.extend(args);
    let out = quiet_command(&program)
        .args(&full)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    // last JSON line is the payload (progress lines may precede it)
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{'));
    match line.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
        Some(v) => Ok(v),
        None => Err(format!(
            "edit tool produced no JSON: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(400).collect::<String>()
        )),
    }
}

#[tauri::command]
fn run_edit_render(app: AppHandle, job_id: String, clip: u32) -> Result<(), String> {
    let (program, base_args) = pipeline_invocation();
    std::thread::spawn(move || {
        let mut args = base_args.clone();
        args.push("--jsonl".to_string());
        args.push("edit".to_string());
        args.push("render-clip".to_string());
        args.push(job_id);
        args.push(clip.to_string());
        stream_pipeline(&app, &program, &args);
    });
    Ok(())
}

#[tauri::command]
fn save_clip_edits(job_id: String, edits: Value) -> Result<(), String> {
    let path = home_dir().join("jobs").join(&job_id).join("clip_edits.json");
    // Merge: the app sends one clip's state at a time; other clips' edits
    // must survive.
    let mut current: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    if let (Some(obj), Some(new)) = (current.as_object_mut(), edits.as_object()) {
        for (k, v) in new {
            obj.insert(k.clone(), v.clone());
        }
    }
    fs::write(&path, serde_json::to_string_pretty(&current).unwrap()).map_err(|e| e.to_string())
}

#[tauri::command]
fn ig_status() -> Result<Value, String> {
    let path = home_dir().join("instagram.json");
    let connected = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    match connected {
        Some(v) => Ok(json!({
            "connected": true,
            "username": v["username"],
            "obtained_at": v["token_obtained_at"],
        })),
        None => Ok(json!({"connected": false})),
    }
}

/// Runs the CLI's OAuth dance (it opens the browser + catches the localhost
/// callback). Blocking by design — the frontend shows a "finish in your
/// browser" state until this returns.
#[tauri::command]
async fn ig_connect(app_id: String, app_secret: String) -> Result<String, String> {
    let (program, base_args) = pipeline_invocation();
    let mut args = base_args;
    args.extend([
        "ig".into(), "connect".into(),
        "--app-id".into(), app_id,
        "--app-secret".into(), app_secret,
    ]);
    let out = quiet_command(&program)
        .args(&args)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if out.status.success() {
        Ok(stdout)
    } else {
        Err(if stderr.is_empty() { stdout } else { stderr })
    }
}

/// One-shot `publikclip ig <args...>` call returning the CLI's JSON line
/// (sync / overview / link / unlink / reject — same contract as edit_tool).
#[tauri::command]
async fn ig_tool(args: Vec<String>) -> Result<Value, String> {
    checked_subcommand(&args, IG_SUBCOMMANDS, "ig")?;
    let (program, base_args) = pipeline_invocation();
    let mut full = base_args;
    full.push("ig".to_string());
    full.extend(args);
    let out = quiet_command(&program)
        .args(&full)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{'));
    match line.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
        Some(v) => Ok(v),
        None => Err(format!(
            "ig tool produced no JSON: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(400).collect::<String>()
        )),
    }
}

#[tauri::command]
fn export_clip(path: String, title: Option<String>) -> Result<String, String> {
    let src = PathBuf::from(&path);
    if !src.exists() {
        return Err("clip file missing".into());
    }
    let downloads = dirs_home().join("Downloads");
    let stem = title.unwrap_or_else(|| "publikclip".into());
    let safe: String = stem
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' { c } else { '_' })
        .collect::<String>()
        .trim()
        .replace(' ', "-")
        .chars()
        .take(60)
        .collect();
    let mut dest = downloads.join(format!("{safe}.mp4"));
    let mut n = 1;
    while dest.exists() {
        dest = downloads.join(format!("{safe}-{n}.mp4"));
        n += 1;
    }
    fs::copy(&src, &dest).map_err(|e| e.to_string())?;
    Ok(dest.to_string_lossy().to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            run_job,
            resume_job,
            job_results,
            list_job_dirs,
            save_gemini_key,
            save_provider_key,
            get_setup_state,
            mark_onboarded,
            check_ollama,
            ig_status,
            ig_connect,
            ig_tool,
            edit_tool,
            run_edit_render,
            save_clip_edits,
            export_clip
        ])
        .setup(|app| {
            let _ = app.get_webview_window("main");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running publikclip");
}
