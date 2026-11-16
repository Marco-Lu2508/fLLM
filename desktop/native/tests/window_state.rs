//! Row 47 of PARITY.md: the window comes back where it was and as big as it was (`window-state` of the old
//! shell), a second start brings the window to the front, and the card *Updates* of Settings > Desktop app.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::desktop::{Desktop, WindowState};
use fllm_app::desktopcards::DkMsg;
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

#[test]
fn the_card_asks_for_updates_and_shows_what_it_found() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    let v = env!("CARGO_PKG_VERSION");
    let mut ui = page(&app);
    for words in ["Updates", "Check for updates", "Look for updates at the start"] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    assert!(ui.find(format!("This is version {v}.")).is_ok());
    assert!(ui.find("Open the download page").is_err(), "nothing is there yet");
    drop(ui);
    let _ = app.update(Message::Dk(DkMsg::UpdateCheck));
    assert_eq!(app.dk.update, State::Checking);
    assert!(page(&app).find("Looking for a newer version ...").is_ok());
    let newer = Release { tag: "v99.0.0".into(), url: "https://github.com/x/y/releases/tag/v99.0.0".into(), version: (99, 0, 0) };
    let _ = app.update(Message::Dk(DkMsg::Updated(Ok(Some(newer)))));
    let mut ui = page(&app);
    assert!(ui.find(format!("Version 99.0.0 is there; you have {v}.")).is_ok());
    assert!(ui.find("Open the download page").is_ok());
    drop(ui);
    let same = Release { tag: format!("v{v}"), url: "u".into(), version: updates::parse_version(v).unwrap() };
    let _ = app.update(Message::Dk(DkMsg::Updated(Ok(Some(same)))));
    assert!(page(&app).find(format!("You have the newest version ({v}).")).is_ok());
    let _ = app.update(Message::Dk(DkMsg::Updated(Err("Could not look for updates: GitHub answered 403".into()))));
    assert!(page(&app).find("Could not look for updates: GitHub answered 403").is_ok());
    // the switch is kept
    assert!(!app.dk.s.check_updates);
    let _ = app.update(Message::Dk(DkMsg::UpdateAuto(true)));
    assert!(Desktop::load(&app.dk.file).check_updates);
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
