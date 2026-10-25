/// The commands the Studio page may call (see `capabilities/studio-page.json`). Each one gets a permission `allow-<name>`.
const PAGE_COMMANDS: &[&str] = &["get_desktop_settings", "set_keep_in_tray", "set_autostart", "open_log_folder", "restart_server", "set_quick_entry", "set_variant"];

fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(PAGE_COMMANDS)))
        .expect("failed to run the Tauri build script");
}
