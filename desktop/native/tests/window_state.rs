//! Row 47 of PARITY.md: the window comes back where it was and as big as it was (`window-state` of the old
//! shell), a second start brings the window to the front, and the card *Updates* of Settings > Desktop app.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::desktop::{Desktop, WindowState};
use fllm_app::desktopcards::{DkMsg, UpdEvent};
use fllm_app::selfupdate;
use fllm_app::store::Store;
use fllm_app::updates::{self, Release, State};
use iced_test::Simulator;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    (app, dir)
}

#[test]
fn only_a_sane_window_is_brought_back() {
    let w = |width, height, x, y| WindowState { width, height, x, y };
    assert_eq!(w(1000.0, 700.0, Some(40.0), Some(30.0)).usable(), Some(w(1000.0, 700.0, Some(40.0), Some(30.0))));
    assert_eq!(w(100.0, 700.0, None, None).usable(), None, "a sliver of a window");
    assert_eq!(w(f32::NAN, 700.0, None, None).usable(), None);
    assert_eq!(w(1000.0, 99999.0, None, None).usable(), None);
    // a place far outside every screen is dropped, the size is kept
    assert_eq!(w(1000.0, 700.0, Some(900_000.0), Some(10.0)).usable(), Some(w(1000.0, 700.0, None, None)));
    assert_eq!(w(1000.0, 700.0, Some(10.0), None).usable(), Some(w(1000.0, 700.0, None, None)), "half a place is none");
    // a screen to the left of the main one has negative places
    assert_eq!(w(1000.0, 700.0, Some(-1500.0), Some(0.0)).usable().unwrap().x, Some(-1500.0));
    let (size, position) = w(1000.0, 700.0, Some(40.0), Some(30.0)).settings();
    assert_eq!((size.width, size.height), (1000.0, 700.0));
    assert!(matches!(position, iced::window::Position::Specific(p) if p.x == 40.0 && p.y == 30.0));
    assert!(matches!(w(1000.0, 700.0, None, None).settings().1, iced::window::Position::Default));
}

#[test]
fn the_window_is_written_when_it_stops_changing_and_when_it_closes() {
    let (mut app, _dir) = app();
    let file = app.dk.file.clone();
    assert_eq!(Desktop::load(&file).window, None);
    let _ = app.update(Message::WinResized(iced::Size::new(1100.0, 760.0)));
    let _ = app.update(Message::WinMoved(iced::Point::new(55.0, 66.0)));
    let _ = app.update(Message::WinResized(iced::Size::new(1120.0, 780.0)));
    // the saving of the first changes comes late: only the last one writes
    let _ = app.update(Message::WinSave(1));
    let _ = app.update(Message::WinSave(2));
    assert_eq!(Desktop::load(&file).window, None, "an old change is not written");
    let _ = app.update(Message::WinSave(3));
    assert_eq!(Desktop::load(&file).window, Some(WindowState { width: 1120.0, height: 780.0, x: Some(55.0), y: Some(66.0) }));
    // a close writes at once
    let _ = app.update(Message::WinResized(iced::Size::new(900.0, 600.0)));
    assert_eq!(Desktop::load(&file).window.unwrap().width, 1120.0);
    let _ = app.update(Message::CloseRequested(iced::window::Id::unique()));
    assert_eq!(Desktop::load(&file).window.unwrap().width, 900.0);
    // a window that is a sliver is not kept
    let _ = app.update(Message::WinResized(iced::Size::new(10.0, 10.0)));
    let _ = app.update(Message::CloseRequested(iced::window::Id::unique()));
    assert_eq!(Desktop::load(&file).window.unwrap().width, 900.0);
}

#[test]
fn a_second_start_brings_the_window_to_the_front() {
    let (mut app, _dir) = app();
    assert_eq!(app.update(Message::InstanceTick).units(), 0, "nobody is watching");
    let flag = Arc::new(AtomicBool::new(false));
    app.watch_instance(flag.clone());
    assert_eq!(app.update(Message::InstanceTick).units(), 0, "nobody asked");
    flag.store(true, Ordering::SeqCst);
    assert!(app.update(Message::InstanceTick).units() >= 1, "the window comes to the front");
    assert!(!flag.load(Ordering::SeqCst), "and the ask is answered once");
}

fn page(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1300.0, 20000.0), app.view())
}

fn release(tag: &str, assets: &[&str]) -> Release {
    Release {
        tag: tag.into(),
        url: format!("https://github.com/x/y/releases/tag/{tag}"),
        version: updates::parse_version(tag).unwrap(),
        assets: assets.iter().map(|n| updates::Asset { name: n.to_string(), url: format!("http://127.0.0.1:9/{n}"), size: 1000 }).collect(),
    }
}

fn updated(app: &mut App, auto: bool, r: Option<Release>) {
    let _ = app.update(Message::Dk(DkMsg::Updated { auto, result: Ok(r) }));
}

#[test]
fn the_card_asks_for_updates_and_shows_what_it_found() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    let v = env!("CARGO_PKG_VERSION");
    let mut ui = page(&app);
    for words in ["Updates", "Check for updates", "Update automatically", "Installed build"] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    assert!(ui.find(format!("This is version {v}.")).is_ok());
    assert!(ui.find("standard Updates fetch the bundle of this build.").is_ok(), "the build of this installation");
    assert!(ui.find("Open the download page").is_err() && ui.find("Download and install").is_err(), "nothing is there yet");
    drop(ui);
    let _ = app.update(Message::Dk(DkMsg::UpdateCheck { auto: false }));
    assert_eq!(app.dk.update, State::Checking);
    assert!(page(&app).find("Looking for a newer version ...").is_ok());
    // a manual look finds a newer one and waits for you
    updated(&mut app, false, Some(release("v99.0.0", &[])));
    let mut ui = page(&app);
    assert!(ui.find(format!("Version 99.0.0 is there; you have {v}.")).is_ok());
    assert!(ui.find("Download and install").is_ok() && ui.find("Open the download page").is_ok());
    drop(ui);
    updated(&mut app, false, Some(release(&format!("v{v}"), &[])));
    assert!(page(&app).find(format!("You have the newest version ({v}).")).is_ok());
    let _ = app.update(Message::Dk(DkMsg::Updated { auto: false, result: Err("Could not look for updates: GitHub answered 403".into()) }));
    assert!(page(&app).find("Could not look for updates: GitHub answered 403").is_ok());
    // the switch is kept; it is on until you turn it off
    assert!(app.dk.s.check_updates);
    let _ = app.update(Message::Dk(DkMsg::UpdateAuto(false)));
    assert!(!Desktop::load(&app.dk.file).check_updates);
}

#[test]
fn a_release_without_the_bundle_of_this_build_says_so_and_fetches_nothing() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    // the build of this test program is the standard one: a release with only a CUDA bundle is no match
    let os = std::env::consts::OS;
    let only_cuda = selfupdate::asset_name(os, std::env::consts::ARCH, selfupdate::Variant::Cuda);
    let names: Vec<String> = only_cuda.into_iter().collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    updated(&mut app, true, Some(release("v99.0.0", &refs)));
    let State::Failed(e) = app.dk.update.clone() else { panic!("{:?}", app.dk.update) };
    assert!(e.contains("Version 99.0.0 has no standard bundle") || e.contains("no ready-made"), "{e}");
    let mut ui = page(&app);
    assert!(ui.find("Open the download page").is_ok(), "the page of the release is still offered");
}

#[test]
fn the_app_of_the_installation_fetches_unpacks_and_waits_for_the_restart() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (mut app, dir) = app();
    let name = selfupdate::asset_name(std::env::consts::OS, std::env::consts::ARCH, selfupdate::Variant::Base);
    let Some(name) = name else { return };
    // a bundle of this system, served by a server of this test
    let files: [(&str, &[u8]); 2] = [("fllm-app", b"new app"), ("fllm-rs", b"new engine")];
    let bytes = if name.ends_with(".zip") {
        use std::io::Write;
        let mut out = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut out);
            for (n, data) in files {
                zip.start_file(n, zip::write::SimpleFileOptions::default()).unwrap();
                zip.write_all(data).unwrap();
            }
            zip.finish().unwrap();
        }
        out.into_inner()
    } else {
        let mut out = Vec::new();
        {
            let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(&mut out, flate2::Compression::fast()));
            for (n, data) in files {
                let mut h = tar::Header::new_gnu();
                h.set_size(data.len() as u64);
                h.set_mode(0o755);
                h.set_cksum();
                tar.append_data(&mut h, n, data).unwrap();
            }
            tar.into_inner().unwrap().finish().unwrap();
        }
        out
    };
    use sha2::Digest;
    let sum = format!("{:x}  {name}\n", sha2::Sha256::digest(&bytes));
    let base = rt.block_on(serve_files(vec![(format!("/{name}"), bytes.clone()), (format!("/{name}.sha256"), sum.clone().into_bytes())]));
    let mut r = release("v99.0.0", &[&name, &format!("{name}.sha256")]);
    for a in &mut r.assets {
        a.url = format!("{base}/{}", a.name);
    }
    let _ = app.update(Message::Go(View::Settings));
    updated(&mut app, false, Some(r.clone()));
    assert_eq!(app.dk.update, State::Available, "found, not fetched yet");
    // run the fetching as the card does and feed what it reports back, as the window would
    let plan = selfupdate::plan(&r, std::env::consts::OS, std::env::consts::ARCH, selfupdate::Variant::Base).unwrap();
    let root = app.dk.stage_root();
    let mut phases = Vec::new();
    let staged = rt.block_on(selfupdate::stage(&plan, &root, "99.0.0", |p| phases.push(p))).unwrap();
    assert!(phases.contains(&selfupdate::Phase::Verify) && phases.contains(&selfupdate::Phase::Unpack));
    assert!(matches!(phases.first(), Some(selfupdate::Phase::Download { .. })));
    let _ = app.update(Message::Dk(DkMsg::UpdateEvent(UpdEvent::Done(Ok(staged.clone())))));
    assert_eq!(app.dk.update, State::Ready);
    let v = env!("CARGO_PKG_VERSION");
    let mut ui = page(&app);
    assert!(ui.find(format!("Version 99.0.0 is ready. Restart to use it; you have {v}.")).is_ok());
    assert!(ui.find("Restart to update").is_ok());
    drop(ui);
    // the restart: the files go over the installation, nothing is started in a test
    let install = dir.path().join("install");
    std::fs::create_dir_all(&install).unwrap();
    std::fs::write(install.join("fllm-app"), "old app").unwrap();
    std::fs::write(install.join("fllm-rs"), "old engine").unwrap();
    let _ = app.install_update(&staged, &install, "fllm-app", false);
    assert_eq!(std::fs::read_to_string(install.join("fllm-app")).unwrap(), "new app");
    assert_eq!(std::fs::read_to_string(install.join("fllm-rs")).unwrap(), "new engine");
    assert!(app.dk.stage.is_none());
    // a copy that is whole and waits is found again without a new download
    let again = selfupdate::stage(&plan, &root, "99.0.0", |_| {});
    let staged = rt.block_on(again).unwrap();
    updated(&mut app, true, Some(r.clone()));
    assert_eq!(app.dk.update, State::Ready, "the unpacked copy of an earlier look is used");
    assert_eq!(app.dk.stage.as_deref(), Some(staged.as_path()));
    // a wrong checksum throws the download away
    let bad = rt.block_on(serve_files(vec![(format!("/{name}"), bytes), (format!("/{name}.sha256"), format!("{}  {name}\n", "0".repeat(64)).into_bytes())]));
    let mut r2 = r;
    for a in &mut r2.assets {
        a.url = format!("{bad}/{}", a.name);
    }
    let plan = selfupdate::plan(&r2, std::env::consts::OS, std::env::consts::ARCH, selfupdate::Variant::Base).unwrap();
    let err = rt.block_on(selfupdate::stage(&plan, &dir.path().join("other"), "99.0.0", |_| {})).unwrap_err();
    assert!(err.contains("does not match the checksum"), "{err}");
    assert!(!dir.path().join("other").join("99.0.0").exists() && !dir.path().join("other").join(&name).exists());
}

async fn serve_files(files: Vec<(String, Vec<u8>)>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = listener.accept().await.unwrap();
            let files = files.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 4096];
                let n = s.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
                let (status, body) = match files.iter().find(|(p, _)| *p == path) {
                    Some((_, b)) => ("200 OK", b.clone()),
                    None => ("404 Not Found", Vec::new()),
                };
                let _ = s.write_all(format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len()).as_bytes()).await;
                let _ = s.write_all(&body).await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn the_look_reads_the_list_of_releases() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let mut n = 0;
        loop {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = s.read(&mut buf).await;
            n += 1;
            let (status, body) = if n == 1 { ("200 OK", r#"[{"tag_name":"v0.2.2","html_url":"https://x/a","draft":false},{"tag_name":"v0.3.0","html_url":"https://x/b","draft":false}]"#) } else { ("403 Forbidden", "{}") };
            let _ = s.write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
        }
    });
    let url = format!("http://{addr}/releases");
    let found = updates::check(&url).await.unwrap().unwrap();
    assert_eq!((found.tag.as_str(), found.url.as_str()), ("v0.3.0", "https://x/b"));
    let err = updates::check(&url).await.unwrap_err();
    assert_eq!(err, "Could not look for updates: GitHub answered 403");
}
