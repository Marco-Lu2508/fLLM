//! The Windows system calls of the sandbox: thin glue that applies what `agent_sandbox_win` decided.
//!
//! NOT VERIFIED ON A REAL WINDOWS MACHINE. The code is cross-compiled and lint-checked for `x86_64-pc-windows-msvc`, and the
//! decisions it applies are unit-tested, but nobody has run it on Windows yet. That is why [`self_test`] exists: before fLLM
//! calls the sandbox available it starts real commands through the launcher on the machine at hand and demands that a write
//! into the allowed folder works and a write outside is refused.
//!
//! Two sides:
//! - the parent (the server) labels the workspace low once, creates the private temp folder and builds the command
//!   `fllm-rs sandbox-exec ...` ([`prepare_workspace`], [`self_test`]);
//! - the launcher ([`run_launcher`], the hidden `sandbox-exec` subcommand) builds the restricted low-integrity token, the job
//!   object and runs the command line from there, forwarding its standard handles and its exit code.
use super::agent_sandbox::Mode;
use super::agent_sandbox_win::{self as win, Policy};
use std::ffi::c_void;
use std::mem::{size_of, zeroed};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr::{null, null_mut};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, LocalFree, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Security::Authorization::ConvertStringSidToSidW;
use windows_sys::Win32::Security::{
    CreateRestrictedToken, GetLengthSid, GetSidIdentifierAuthority, GetSidSubAuthority, GetSidSubAuthorityCount, GetTokenInformation,
    SetTokenInformation, TokenGroups, TokenIntegrityLevel, DISABLE_MAX_PRIVILEGE, PSID, SID_AND_ATTRIBUTES,
    TOKEN_ADJUST_DEFAULT, TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_GROUPS, TOKEN_MANDATORY_LABEL, TOKEN_QUERY,
};
use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectBasicUIRestrictions, JobObjectExtendedLimitInformation, SetInformationJobObject,
    JOBOBJECT_BASIC_UI_RESTRICTIONS, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_UILIMIT_DESKTOP, JOB_OBJECT_UILIMIT_DISPLAYSETTINGS, JOB_OBJECT_UILIMIT_EXITWINDOWS,
    JOB_OBJECT_UILIMIT_GLOBALATOMS, JOB_OBJECT_UILIMIT_HANDLES, JOB_OBJECT_UILIMIT_READCLIPBOARD, JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS,
    JOB_OBJECT_UILIMIT_WRITECLIPBOARD,
};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::SystemServices::SE_GROUP_INTEGRITY;
use windows_sys::Win32::System::Threading::{
    CreateProcessAsUserW, GetCurrentProcess, GetExitCodeProcess, OpenProcessToken, ResumeThread, TerminateProcess, WaitForSingleObject,
    CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, INFINITE, PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOW,
};

/// Low mandatory integrity level (`S-1-16-4096`).
const LOW_INTEGRITY_SID: &str = "S-1-16-4096";

/// An owned Windows handle, closed when it goes out of scope.
struct Handle(HANDLE);

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
            // SAFETY: the handle is owned by this value and closed exactly once.
            unsafe { CloseHandle(self.0) };
        }
    }
}

fn failure(what: &str) -> String {
    // SAFETY: reads the calling thread's last error, no pointers involved.
    format!("{what} failed (Windows error {})", unsafe { GetLastError() })
}

fn system32(program: &str) -> PathBuf {
    let root = std::env::var_os("SystemRoot").or_else(|| std::env::var_os("windir")).unwrap_or_else(|| "C:\\Windows".into());
    Path::new(&root).join("System32").join(program)
}

/// A path without the `\\?\` prefix the standard library adds when it canonicalises (cmd and icacls do not like it).
fn plain(path: &Path) -> String {
    let text = path.to_string_lossy();
    match text.strip_prefix("\\\\?\\") {
        Some(rest) if !rest.starts_with("UNC\\") => rest.to_owned(),
        _ => text.into_owned(),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the parent: workspace, temp folder, self-test

fn user_profile() -> Result<String, String> {
    std::env::var("USERPROFILE").map_err(|_| "USERPROFILE is not set, so the low-integrity folder cannot be found".to_owned())
}

/// The private temp folder of this server run (created when missing). Folders of earlier runs older than two days are removed.
fn private_temp() -> Result<PathBuf, String> {
    let base = win::private_temp_dir(&win::local_low_dir(&user_profile()?), "x");
    let parent = base.parent().map(Path::to_path_buf).ok_or("no parent folder")?;
    let _ = std::fs::create_dir_all(&parent);
    if let Ok(entries) = std::fs::read_dir(&parent) {
        let limit = Duration::from_secs(2 * 24 * 3600);
        for entry in entries.flatten() {
            let old = entry.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age > limit);
            if old {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    let temp = parent.join(std::process::id().to_string());
    std::fs::create_dir_all(&temp).map_err(|e| format!("cannot create the private temp folder {}: {e}", temp.display()))?;
    Ok(temp)
}

fn labelled_file() -> PathBuf {
    crate::hub::studio_data_file("sandbox-labelled.txt")
}

/// Gives the workspace the low integrity label (once per folder, remembered in `sandbox-labelled.txt`) and returns the
/// private temp folder. Refuses folders that must not be labelled.
pub(super) fn prepare_workspace(root: &Path) -> Result<PathBuf, String> {
    static DONE: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    let root_text = plain(root);
    win::workspace_label_allowed(&root_text, &user_profile()?)?;
    let done = DONE.get_or_init(|| Mutex::new(Vec::new()));
    let mut done = done.lock().unwrap_or_else(|e| e.into_inner());
    let key = root_text.to_lowercase();
    if !done.contains(&key) {
        let listed = std::fs::read_to_string(labelled_file()).map(|text| win::labelled_contains(&text, &root_text)).unwrap_or(false);
        if !listed {
            let output = Command::new(system32("icacls.exe"))
                .args(win::icacls_label_args(&root_text))
                .stdin(Stdio::null())
                .creation_flags(CREATE_NO_WINDOW)
                .output()
                .map_err(|e| format!("cannot run icacls to label the workspace: {e}"))?;
            if !output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                return Err(format!(
                    "the sandbox could not label the workspace {root_text} as writable for sandboxed commands (icacls: {})",
                    text.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("no message").trim()
                ));
            }
            let path = labelled_file();
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut text = std::fs::read_to_string(&path).unwrap_or_default();
            text.push_str(&root_text);
            text.push('\n');
            let _ = std::fs::write(&path, text);
        }
        done.push(key);
    }
    private_temp()
}

/// The arguments (after the program) that run `line` through the launcher.
pub(super) fn launcher_command(mode: Mode, line: &str, root: &Path, cwd: &Path, temp: &Path) -> Vec<String> {
    win::launcher_args(&win::LauncherRequest { mode, root: plain(root), cwd: plain(cwd), temp: plain(temp), line: line.to_owned() })
}

/// Starts the sandbox for real and checks what it does. Returns the launcher program on success.
pub(super) fn self_test() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find the fLLM program to use as launcher: {e}"))?;
    let temp = private_temp()?;
    let allowed = temp.join("probe-allowed.txt");
    let denied = std::env::temp_dir().join(format!("fllm-sandbox-probe-{}.txt", std::process::id()));
    let run = |line: &str| -> Result<bool, String> {
        let mut child = Command::new(&exe)
            .args(launcher_command(Mode::Workspace, line, &temp, &temp, &temp))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("cannot start the sandbox launcher: {e}"))?;
        let started = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status.success()),
                Ok(None) if started.elapsed() > Duration::from_secs(20) => {
                    let _ = child.kill();
                    return Err("the sandbox launcher did not finish within 20 seconds".to_owned());
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(25)),
                Err(e) => return Err(e.to_string()),
            }
        }
    };
    // `echo x> file` fails with a non-zero exit code when the file cannot be created
    let write_line = |path: &str| format!("echo x> \"{path}\"");
    let result = win::enforcement_selftest(&run, &write_line, &plain(&allowed), &plain(&denied));
    let _ = std::fs::remove_file(&allowed);
    let _ = std::fs::remove_file(&denied);
    result.map(|()| exe)
}

// ---------------------------------------------------------------------------------------------------------------------
// the launcher

/// `fllm-rs sandbox-exec ...`: runs one command line under the policy and returns its exit code.
pub fn run_launcher(args: &[String]) -> i32 {
    match launch(args) {
        Ok(code) => code as i32,
        Err(message) => {
            // the command's stderr is the agent's window into this: say what happened, in plain words
            eprintln!("fLLM sandbox: {message}");
            125
        }
    }
}

fn launch(args: &[String]) -> Result<u32, String> {
    let request = win::parse_launcher_args(args)?;
    let policy = win::policy_for(request.mode)?;
    let token = sandbox_token(&policy)?;
    let job = make_job(&policy)?;
    let env = win::scrubbed_env(std::env::vars(), &request.temp, policy.scrub_environment);
    let environment = win::environment_block(&env);
    let comspec = system32("cmd.exe");
    let mut command_line = win::wide(&win::cmd_command_line(&comspec.to_string_lossy(), &request.line)?);
    let cwd = win::wide(&request.cwd);

    // SAFETY: plain C structs, all-zero is their documented initial state.
    let mut startup: STARTUPINFOW = unsafe { zeroed() };
    startup.cb = size_of::<STARTUPINFOW>() as u32;
    startup.dwFlags = STARTF_USESTDHANDLES;
    // SAFETY: GetStdHandle only reads the process's standard handles.
    unsafe {
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    }
    // SAFETY: as above.
    let mut info: PROCESS_INFORMATION = unsafe { zeroed() };
    // SAFETY: every pointer is valid for the call: the token and job are owned handles, the strings are NUL-terminated UTF-16
    // buffers that live until the call returns (the command line is mutable as the API demands), the environment block is
    // double-NUL terminated and flagged as Unicode.
    let created = unsafe {
        CreateProcessAsUserW(
            token.0,
            null(),
            command_line.as_mut_ptr(),
            null(),
            null(),
            1,
            CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
            environment.as_ptr().cast::<c_void>(),
            cwd.as_ptr(),
            &startup,
            &mut info,
        )
    };
    if created == 0 {
        return Err(failure("starting the command under the restricted token"));
    }
    let process = Handle(info.hProcess);
    let thread = Handle(info.hThread);
    // SAFETY: both handles are valid and owned here.
    unsafe {
        if AssignProcessToJobObject(job.0, process.0) == 0 {
            let message = failure("putting the command into its job object");
            TerminateProcess(process.0, 1);
            return Err(message);
        }
        if ResumeThread(thread.0) == u32::MAX {
            let message = failure("starting the command");
            TerminateProcess(process.0, 1);
            return Err(message);
        }
        WaitForSingleObject(process.0, INFINITE);
        let mut code = 0u32;
        if GetExitCodeProcess(process.0, &mut code) == 0 {
            return Err(failure("reading the exit code"));
        }
        // dropping `job` (end of this function) kills whatever the command left behind
        Ok(code)
    }
}

/// The token the command runs with: the launcher's own, minus privileges and administrator-like groups, at low integrity.
fn sandbox_token(policy: &Policy) -> Result<Handle, String> {
    // SAFETY: standard token calls; every out-pointer refers to a local that outlives the call, and the SID buffers stay
    // alive until the restricted token exists.
    unsafe {
        let mut own: HANDLE = null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT, &mut own) == 0 {
            return Err(failure("opening the process token"));
        }
        let own = Handle(own);
        let groups = if policy.drop_privileges { groups_to_disable(own.0)? } else { GroupBuffer::empty() };
        let list = groups.list();
        let flags = if policy.drop_privileges { DISABLE_MAX_PRIVILEGE } else { 0 };
        let mut restricted: HANDLE = null_mut();
        let count = u32::try_from(list.len()).unwrap_or(0);
        let disable = if list.is_empty() { null() } else { list.as_ptr() };
        if CreateRestrictedToken(own.0, flags, count, disable, 0, null(), 0, null(), &mut restricted) == 0 {
            return Err(failure("creating the restricted token"));
        }
        let restricted = Handle(restricted);
        if policy.low_integrity {
            let mut sid: PSID = null_mut();
            if ConvertStringSidToSidW(win::wide(LOW_INTEGRITY_SID).as_ptr(), &mut sid) == 0 {
                return Err(failure("building the low integrity SID"));
            }
            let label = TOKEN_MANDATORY_LABEL { Label: SID_AND_ATTRIBUTES { Sid: sid, Attributes: SE_GROUP_INTEGRITY as u32 } };
            let size = size_of::<TOKEN_MANDATORY_LABEL>() as u32 + GetLengthSid(sid);
            let ok = SetTokenInformation(restricted.0, TokenIntegrityLevel, (&label as *const TOKEN_MANDATORY_LABEL).cast::<c_void>(), size);
            let message = if ok == 0 { Some(failure("lowering the integrity level of the token")) } else { None };
            LocalFree(sid);
            if let Some(message) = message {
                return Err(message);
            }
        }
        Ok(restricted)
    }
}

/// The token's groups that [`win::should_disable_group`] picks, kept together with the buffer their SIDs point into.
struct GroupBuffer {
    _buffer: Vec<u64>,
    chosen: Vec<SID_AND_ATTRIBUTES>,
}

impl GroupBuffer {
    fn empty() -> Self {
        Self { _buffer: Vec::new(), chosen: Vec::new() }
    }
    fn list(&self) -> &[SID_AND_ATTRIBUTES] {
        &self.chosen
    }
}

/// # Safety
/// `token` has to be a valid token handle opened with `TOKEN_QUERY`.
unsafe fn groups_to_disable(token: HANDLE) -> Result<GroupBuffer, String> {
    let mut needed = 0u32;
    GetTokenInformation(token, TokenGroups, null_mut(), 0, &mut needed);
    if needed == 0 {
        return Err(failure("measuring the token groups"));
    }
    // u64 elements: the buffer is aligned for the pointers inside TOKEN_GROUPS
    let mut buffer = vec![0u64; (needed as usize).div_ceil(8)];
    if GetTokenInformation(token, TokenGroups, buffer.as_mut_ptr().cast::<c_void>(), needed, &mut needed) == 0 {
        return Err(failure("reading the token groups"));
    }
    let groups = &*(buffer.as_ptr().cast::<TOKEN_GROUPS>());
    let entries = std::slice::from_raw_parts(groups.Groups.as_ptr(), groups.GroupCount as usize);
    let mut chosen = Vec::new();
    for entry in entries {
        let sid = entry.Sid;
        let authority = (*GetSidIdentifierAuthority(sid)).Value;
        let count = *GetSidSubAuthorityCount(sid);
        let subs: Vec<u32> = (0..u32::from(count)).map(|i| *GetSidSubAuthority(sid, i)).collect();
        if win::should_disable_group(authority, &subs) {
            chosen.push(SID_AND_ATTRIBUTES { Sid: sid, Attributes: 0 });
        }
    }
    Ok(GroupBuffer { _buffer: buffer, chosen })
}

fn physical_memory() -> Result<u64, String> {
    // SAFETY: MEMORYSTATUSEX is a plain struct whose length field has to be set before the call.
    unsafe {
        let mut status: MEMORYSTATUSEX = zeroed();
        status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) == 0 {
            return Err(failure("reading the physical memory size"));
        }
        Ok(status.ullTotalPhys)
    }
}

/// The job object of one command: dies with the launcher, limits processes and memory, closes the window-system doors.
fn make_job(policy: &Policy) -> Result<Handle, String> {
    // SAFETY: the structs are plain data passed by pointer with their exact size; the job handle is owned.
    unsafe {
        let job = Handle(CreateJobObjectW(null(), null()));
        if job.0.is_null() {
            return Err(failure("creating the job object"));
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        let mut flags = 0;
        if policy.kill_on_close {
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        }
        if let Some(count) = policy.max_processes {
            flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            limits.BasicLimitInformation.ActiveProcessLimit = count;
        }
        if let Some(fraction) = policy.memory_fraction {
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            limits.JobMemoryLimit = usize::try_from(win::memory_limit_bytes(physical_memory()?, fraction)).unwrap_or(usize::MAX);
        }
        limits.BasicLimitInformation.LimitFlags = flags;
        let size = size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
        if SetInformationJobObject(job.0, JobObjectExtendedLimitInformation, (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(), size) == 0 {
            return Err(failure("setting the job limits"));
        }
        if policy.ui_limits {
            let ui = JOBOBJECT_BASIC_UI_RESTRICTIONS {
                UIRestrictionsClass: JOB_OBJECT_UILIMIT_DESKTOP
                    | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
                    | JOB_OBJECT_UILIMIT_EXITWINDOWS
                    | JOB_OBJECT_UILIMIT_GLOBALATOMS
                    | JOB_OBJECT_UILIMIT_HANDLES
                    | JOB_OBJECT_UILIMIT_READCLIPBOARD
                    | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
                    | JOB_OBJECT_UILIMIT_WRITECLIPBOARD,
            };
            let size = size_of::<JOBOBJECT_BASIC_UI_RESTRICTIONS>() as u32;
            if SetInformationJobObject(job.0, JobObjectBasicUIRestrictions, (&ui as *const JOBOBJECT_BASIC_UI_RESTRICTIONS).cast::<c_void>(), size) == 0 {
                return Err(failure("setting the job UI restrictions"));
            }
        }
        Ok(job)
    }
}
