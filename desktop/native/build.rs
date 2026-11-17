//! Puts the icon of the app into `fllm-app.exe` on Windows (the file in the Explorer, the task bar before the window has its own,
//! the shortcuts that point at it). Other systems have nothing to do here.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=installer/fllm.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        #[cfg(windows)]
        {
            let mut resource = winresource::WindowsResource::new();
            resource.set_icon("installer/fllm.ico");
            resource.set("ProductName", "fLLM");
            resource.set("FileDescription", "fLLM");
            resource.set("CompanyName", "Marco Lutz");
            resource.set("LegalCopyright", "Marco Lutz");
            resource.compile().expect("the icon of the program could not be embedded");
        }
    }
}
