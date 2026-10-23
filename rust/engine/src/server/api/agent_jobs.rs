//! Shell commands of the coding agent: how a command line is started, and the background jobs
//! (`bash` with `background: true`, `bash_output`, `bash_kill`). A job's output goes into a bounded buffer
//! (the oldest bytes are dropped first); every job is killed when the session ends or the server stops.
use std::{
    collections::HashMap,
    io::Read,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const MAX_RUNNING_JOBS: usize = 8;
const MAX_KEPT_JOBS: usize = 32;
const BUFFER_BYTES: usize = 256 * 1024;
const READ_BYTES: usize = 30_000;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The platform shell running one command line.
pub(super) fn shell_command(line: &str) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        c.arg("/V:ON").arg("/S").arg("/C").raw_arg(format!("\"{line}\""));
        c
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new("sh");
        c.arg("-c").arg(line);
        // its own process group, so a whole job can be stopped with everything it started
        c.process_group(0);
        c
    }
}

/// The command followed by a line that reports the final working directory and keeps the exit code.
pub(super) fn with_cwd_report(command: &str, mark: &str) -> String {
    if cfg!(windows) {
        format!("{command} & echo. & echo {mark}!cd! & exit /b !errorlevel!")
    } else {
        format!("{command}\n__fllm_rc=$?\nprintf '\\n{mark}%s\\n' \"$(pwd)\"\nexit $__fllm_rc")
    }
}

#[derive(Default)]
struct Buffer {
    data: Vec<u8>,
    /// How many bytes were dropped from the front: the offset of `data[0]` in the whole output.
    dropped: u64,
}

impl Buffer {
    fn push(&mut self, chunk: &[u8]) {
        self.data.extend_from_slice(chunk);
        if self.data.len() > BUFFER_BYTES {
            let cut = self.data.len() - BUFFER_BYTES;
            self.data.drain(..cut);
            self.dropped += cut as u64;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Exited(Option<i32>),
    Killed,
}

struct Job {
    command: String,
    started: Instant,
    pid: u32,
    buffer: Arc<Mutex<Buffer>>,
    state: Arc<Mutex<State>>,
    child: Arc<Mutex<Child>>,
}

impl Job {
    fn kill(&self) {
        {
            let mut state = lock(&self.state);
            if *state != State::Running {
                return;
            }
            *state = State::Killed;
        }
        kill_tree(self.pid);
        let _ = lock(&self.child).kill();
    }

    fn running(&self) -> bool {
        *lock(&self.state) == State::Running
    }
}

fn kill_tree(pid: u32) {
    #[cfg(windows)]
    let mut c = {
        let mut c = Command::new("taskkill");
        c.args(["/T", "/F", "/PID", &pid.to_string()]);
        c
    };
    #[cfg(not(windows))]
    let mut c = {
        let mut c = Command::new("kill");
        c.args(["-KILL", &format!("-{pid}")]);
        c
    };
    let _ = c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
}

fn pump<R: Read + Send + 'static>(stream: Option<R>, buffer: Arc<Mutex<Buffer>>) {
    std::thread::spawn(move || {
        let Some(mut stream) = stream else { return };
        let mut chunk = [0u8; 8192];
        while let Ok(n) = stream.read(&mut chunk) {
            if n == 0 {
                break;
            }
            lock(&buffer).push(&chunk[..n]);
        }
    });
}

/// The background jobs of one agent session.
#[derive(Default)]
pub(super) struct Jobs {
    jobs: Mutex<HashMap<String, Job>>,
    counter: Mutex<u64>,
}

impl Jobs {
    /// Starts `command` in `dir` and returns a message with the job id.
    pub(super) fn start(&self, command: &str, dir: &Path) -> Result<String, String> {
        let mut jobs = lock(&self.jobs);
        if jobs.values().filter(|j| j.running()).count() >= MAX_RUNNING_JOBS {
            return Err(format!("{MAX_RUNNING_JOBS} background jobs are already running; stop one with bash_kill first"));
        }
        if jobs.len() >= MAX_KEPT_JOBS {
            // forget the oldest finished jobs
            let mut finished: Vec<(String, Instant)> =
                jobs.iter().filter(|(_, j)| !j.running()).map(|(id, j)| (id.clone(), j.started)).collect();
            finished.sort_by_key(|(_, started)| *started);
            for (id, _) in finished.into_iter().take(jobs.len() + 1 - MAX_KEPT_JOBS) {
                jobs.remove(&id);
            }
        }
        let mut child = shell_command(command)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start the command: {e}"))?;
        let buffer = Arc::new(Mutex::new(Buffer::default()));
        pump(child.stdout.take(), buffer.clone());
        pump(child.stderr.take(), buffer.clone());
        let pid = child.id();
        let child = Arc::new(Mutex::new(child));
        let state = Arc::new(Mutex::new(State::Running));
        {
            let (child, state) = (child.clone(), state.clone());
            std::thread::spawn(move || loop {
                let status = lock(&child).try_wait();
                match status {
                    Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                    Ok(Some(status)) => {
                        let mut state = lock(&state);
                        if *state == State::Running {
                            *state = State::Exited(status.code());
                        }
                        break;
                    }
                    Err(_) => {
                        let mut state = lock(&state);
                        if *state == State::Running {
                            *state = State::Exited(None);
                        }
                        break;
                    }
                }
            });
        }
        let id = {
            let mut counter = lock(&self.counter);
            *counter += 1;
            format!("job{}", *counter)
        };
        jobs.insert(id.clone(), Job { command: command.to_owned(), started: Instant::now(), pid, buffer, state, child });
        Ok(format!(
            "Started background job {id}. Read its output with bash_output {{job_id: \"{id}\"}} and stop it with bash_kill."
        ))
    }

    /// The output of a job from byte offset `since` on (as reported by the previous call) and its state.
    pub(super) fn output(&self, id: &str, since: Option<u64>) -> Result<String, String> {
        let jobs = lock(&self.jobs);
        let job = jobs.get(id).ok_or_else(|| format!("there is no background job '{id}'"))?;
        let (text, next, note) = {
            let buffer = lock(&job.buffer);
            let wanted = since.unwrap_or(0);
            let start = wanted.max(buffer.dropped);
            let note = if wanted < buffer.dropped { "(older output was dropped)\n" } else { "" };
            let from = usize::try_from(start - buffer.dropped).unwrap_or(usize::MAX).min(buffer.data.len());
            let mut to = (from + READ_BYTES).min(buffer.data.len());
            // do not cut inside a UTF-8 sequence when more output follows
            while to < buffer.data.len() && to > from && (buffer.data[to] & 0xC0) == 0x80 {
                to -= 1;
            }
            (String::from_utf8_lossy(&buffer.data[from..to]).into_owned(), start + (to - from) as u64, note)
        };
        let status = match *lock(&job.state) {
            State::Running => "running".to_owned(),
            State::Exited(Some(code)) => format!("exited with code {code}"),
            State::Exited(None) => "exited".to_owned(),
            State::Killed => "stopped (killed)".to_owned(),
        };
        Ok(format!(
            "job: {id} ({})\nstatus: {status}\nnext_since: {next}\n--- output ---\n{note}{}",
            job.command.chars().take(80).collect::<String>(),
            if text.is_empty() { "(no new output)\n".to_owned() } else { text }
        ))
    }

    pub(super) fn kill(&self, id: &str) -> Result<String, String> {
        let jobs = lock(&self.jobs);
        let job = jobs.get(id).ok_or_else(|| format!("there is no background job '{id}'"))?;
        if job.running() {
            job.kill();
            Ok(format!("Stopped job {id}."))
        } else {
            Ok(format!("Job {id} had already finished."))
        }
    }

    /// Kills every running job; returns how many were running.
    pub(super) fn kill_all(&self) -> usize {
        let mut jobs = lock(&self.jobs);
        let running = jobs.values().filter(|j| j.running()).count();
        for job in jobs.values() {
            job.kill();
        }
        jobs.clear();
        running
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        self.kill_all();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn wait_for(jobs: &Jobs, id: &str, needle: &str) -> String {
        for _ in 0..100 {
            let out = jobs.output(id, None).unwrap();
            if out.contains(needle) {
                return out;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("'{needle}' never appeared in the output of {id}");
    }

    #[test]
    fn a_job_streams_output_and_reports_its_exit() {
        let jobs = Jobs::default();
        let dir = std::env::temp_dir();
        let started = jobs.start("echo one; sleep 0.3; echo two; exit 4", &dir).unwrap();
        assert!(started.contains("job1"), "{started}");
        let early = wait_for(&jobs, "job1", "one");
        assert!(early.contains("status: running") || early.contains("exited"), "{early}");
        let done = wait_for(&jobs, "job1", "exited with code 4");
        assert!(done.contains("one\ntwo\n"), "{done}");
        // reading from the reported offset returns only what is new
        let next: u64 = done.lines().find_map(|l| l.strip_prefix("next_since: ")).unwrap().parse().unwrap();
        assert!(jobs.output("job1", Some(next)).unwrap().contains("(no new output)"));
        assert!(jobs.output("nope", None).is_err());
    }

    #[test]
    fn a_running_job_can_be_killed_and_the_session_end_kills_the_rest() {
        let jobs = Jobs::default();
        let dir = std::env::temp_dir();
        jobs.start("sleep 30", &dir).unwrap();
        jobs.start("sleep 30", &dir).unwrap();
        assert!(jobs.kill("job1").unwrap().starts_with("Stopped"));
        assert!(jobs.output("job1", None).unwrap().contains("stopped (killed)"));
        assert!(jobs.kill("job1").unwrap().contains("already finished"));
        assert_eq!(jobs.kill_all(), 1);
        assert!(jobs.output("job2", None).is_err(), "ended sessions forget their jobs");
    }

    #[test]
    fn the_buffer_keeps_only_the_newest_bytes() {
        let mut buffer = Buffer::default();
        buffer.push(&vec![b'a'; BUFFER_BYTES]);
        buffer.push(b"tail");
        assert_eq!(buffer.data.len(), BUFFER_BYTES);
        assert_eq!(buffer.dropped, 4);
        assert!(buffer.data.ends_with(b"tail"));
    }
}
