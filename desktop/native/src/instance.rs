//! Row 47 of PARITY.md: one fLLM at a time (`single-instance` of the old shell). A second start does
//! not open a second window: it tells the first one to come to the front and ends.
//!
//! The first one listens on a port of the loopback (the same for the same folder of data, so two
//! installations with their own data do not meet). The second one asks there; only an answer in the
//! words of this program counts, so another program that happens to hold the port is not mistaken
//! for fLLM.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const ASK: &[u8] = b"fllm:show\n";
const ANSWER: &[u8] = b"fllm:ok\n";

/// The port of an installation: the same folder of data always gives the same one.
pub fn port_for(data_dir: &std::path::Path) -> u16 {
    // FNV-1a: no dependency, and the same on every system
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data_dir.to_string_lossy().as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    49_152 + (h % 16_000) as u16
}

/// What a start comes to.
pub enum Role {
    /// Nobody was there: this one runs, and `shown` turns true when another start asked it to come to the front.
    First(Arc<AtomicBool>),
    /// Another fLLM answered and was told to come to the front: this one ends.
    Second,
}

pub fn acquire(port: u16) -> Role {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let listener = match TcpListener::bind(addr) {
        Ok(l) => l,
        Err(_) => {
            // the port is taken: by another fLLM, or by something else
            if tell(addr) {
                return Role::Second;
            }
            // something else holds it: this one runs without the hand-over
            return Role::First(Arc::new(AtomicBool::new(false)));
        }
    };
    let shown = Arc::new(AtomicBool::new(false));
    let flag = shown.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
            let mut buf = [0u8; ASK.len()];
            if stream.read_exact(&mut buf).is_ok() && buf == *ASK {
                let _ = stream.write_all(ANSWER);
                flag.store(true, Ordering::SeqCst);
            }
        }
    });
    Role::First(shown)
}

/// Asks the one that holds the port to come to the front; `true` when it answered like fLLM.
fn tell(addr: SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else { return false };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(800)));
    if stream.write_all(ASK).is_err() {
        return false;
    }
    let mut buf = [0u8; ANSWER.len()];
    stream.read_exact(&mut buf).is_ok() && buf == *ANSWER
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_port() -> u16 {
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap().port()
    }

    #[test]
    fn the_second_start_hands_over_to_the_first() {
        let port = free_port();
        let Role::First(shown) = acquire(port) else { panic!("nobody was there") };
        assert!(!shown.load(Ordering::SeqCst));
        assert!(matches!(acquire(port), Role::Second), "the second start ends");
        for _ in 0..50 {
            if shown.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(shown.swap(false, Ordering::SeqCst), "the first one was told to come to the front");
        assert!(matches!(acquire(port), Role::Second));
    }

    #[test]
    fn a_port_held_by_another_program_does_not_stop_the_start() {
        let other = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = other.local_addr().unwrap().port();
        // the other program does not answer in the words of fLLM
        std::thread::spawn(move || {
            for s in other.incoming().flatten() {
                let mut s = s;
                let _ = s.write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n");
            }
        });
        assert!(matches!(acquire(port), Role::First(_)), "this start goes on, without the hand-over");
    }

    #[test]
    fn the_port_belongs_to_the_folder_of_data() {
        let a = port_for(std::path::Path::new("/home/a/.local/share/fllm"));
        assert_eq!(a, port_for(std::path::Path::new("/home/a/.local/share/fllm")));
        assert_ne!(a, port_for(std::path::Path::new("/home/b/.local/share/fllm")));
        assert!((49_152..65_152).contains(&a));
    }
}
