//! Opening the vendor shared libraries of the NPU backends (package 11; features `npu-xdna` and `npu-qnn`).

use libloading::Library;
use std::path::Path;

/// Opens the shared library at `path` (a file name is resolved by the system loader).
///
/// On Windows an absolute path is loaded with `LOAD_WITH_ALTERED_SEARCH_PATH`, so that the DLLs next to it (the Ryzen AI `deployment`
/// folder, the QAIRT `lib` folder) are found as dependencies without being on `PATH`.
///
/// # Safety
/// Loading a library runs its initializers, exactly like `Library::new`.
pub unsafe fn open(path: &Path) -> Result<Library, libloading::Error> {
    #[cfg(windows)]
    {
        if path.is_absolute() {
            return libloading::os::windows::Library::load_with_flags(
                path,
                libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
            )
            .map(Library::from);
        }
    }
    Library::new(path)
}
