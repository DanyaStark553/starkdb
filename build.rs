#[cfg(windows)]
fn main() {
    let mut res = winres::WindowsResource::new();
    res.set_icon("assets/favicon.ico");
    res.set("FileDescription", "StarkDB // Victory Boost — Cognitive Associative Graph & Vector DB");
    res.set("ProductName", "StarkDB");
    res.set("OriginalFilename", "starkdb.exe");
    res.set("LegalCopyright", "Danya <starkos>");
    if let Err(e) = res.compile() {
        eprintln!("Warning: Failed to compile Windows resource: {}", e);
    }
}

#[cfg(not(windows))]
fn main() {}
