//! Cabinet impulse response browser.
//!
//! A file picker built out of `vizia` views rather than a native dialog:
//! `CLAUDE.md` requires explicit approval before a crate is added, and every
//! cross-platform file dialog is a crate. `std::fs::read_dir` is not, so the
//! browser walks the filesystem itself. That also means it looks like the rest
//! of the faceplate instead of like whichever toolkit the host happens to use.
//!
//! Nothing here runs on the audio thread. Selecting a file decodes it through
//! [`crate::ir`] on the editor thread and hands the taps to the audio thread
//! through [`crate::shared::IrSlot`], which neither side locks.

use std::path::{Path, PathBuf};

use nih_plug_vizia::vizia::prelude::*;

use crate::ir::{self, DirectoryEntry};

// The editor model is imported under another name: vizia's prelude also
// exports a `Data` trait, and the two would otherwise collide.
use super::{Data as EditorData, UiEvent};

/// Label shown when the built-in cabinet is in use.
pub const DEFAULT_CABINET_LABEL: &str = "4x12 BUILT-IN";

/// Longest cabinet name the launcher button shows before eliding.
///
/// The button is 78 logical pixels wide, which fits about sixteen characters
/// of the faceplate's condensed face. Vendor IR filenames routinely run past
/// sixty, so the tail — which is the part that distinguishes `...57-cone.wav`
/// from `...57-cap.wav` — is what gets kept.
const MAX_CABINET_LABEL: usize = 16;

// `Data` is a foreign trait and `DirectoryEntry` is a local type, so this
// implementation belongs to this crate. It is here rather than in
// `crate::ir` so that the loader stays free of any `vizia` dependency.
impl Data for DirectoryEntry {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

/// Shortens a cabinet name to fit the launcher button.
///
/// Keeps the tail, because IR libraries name files by prefixing a common
/// cabinet name onto the part that actually varies.
pub fn short_cabinet_label(name: &str) -> String {
    let characters: Vec<char> = name.chars().collect();
    if characters.len() <= MAX_CABINET_LABEL {
        return name.to_uppercase();
    }
    let tail: String = characters
        .iter()
        .skip(characters.len() - (MAX_CABINET_LABEL - 1))
        .collect();
    format!("…{}", tail.to_uppercase())
}

/// Button caption for the response at `path`, which may be empty.
pub fn cabinet_label(path: &str) -> String {
    if path.is_empty() {
        return DEFAULT_CABINET_LABEL.to_string();
    }
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(short_cabinet_label)
        .unwrap_or_else(|| DEFAULT_CABINET_LABEL.to_string())
}

/// Renders a directory for the path strip above the listing.
///
/// Long paths are elided from the left, keeping the folders nearest the files,
/// which is the part that tells the user where they are.
pub fn display_directory(directory: &Path) -> String {
    const MAX: usize = 58;
    let full = directory.to_string_lossy().replace('\\', "/");
    let characters: Vec<char> = full.chars().collect();
    if characters.len() <= MAX {
        return full;
    }
    let tail: String = characters
        .iter()
        .skip(characters.len() - (MAX - 1))
        .collect();
    format!("…{tail}")
}

impl EditorData {
    /// Lists `directory` and makes it the browser's current location.
    ///
    /// A directory that cannot be read leaves the previous listing in place and
    /// reports why, rather than dropping the user into an empty view with no
    /// explanation.
    pub fn browse_to(&mut self, directory: PathBuf) {
        match ir::list_directory(&directory) {
            Ok(entries) => {
                self.ir_status = if entries.is_empty() {
                    "no folders or WAV files here".to_string()
                } else {
                    String::new()
                };
                self.ir_entries = entries;
                self.ir_directory = display_directory(&directory);
                self.ir_location = directory;
            }
            Err(error) => self.ir_status = format!("cannot open folder: {error}"),
        }
    }

    /// Moves to the parent of the current directory, if there is one.
    pub fn browse_up(&mut self) {
        // `Path::parent` returns `Some("")` for a bare relative name and
        // `None` at a filesystem root; both mean "there is nowhere above here".
        let parent = self
            .ir_location
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf);
        match parent {
            Some(parent) => self.browse_to(parent),
            None => self.ir_status = "already at the top".to_string(),
        }
    }

    /// Acts on the row at `index`: navigates into a folder, or loads a file.
    pub fn select_entry(&mut self, index: usize) {
        let Some(entry) = self.ir_entries.get(index).cloned() else {
            return;
        };
        if entry.is_directory {
            self.browse_to(entry.path);
        } else {
            self.load_cabinet(&entry.path);
        }
    }

    /// Decodes `path` and publishes it to the audio thread.
    ///
    /// The host sample rate arrives through the shared cell that
    /// `Plugin::initialize` writes; before the host has called that there is
    /// nothing to resample to, so the load is refused rather than guessed at.
    pub fn load_cabinet(&mut self, path: &Path) {
        let sample_rate = self.sample_rate.load();
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            self.ir_status = "the host has not started audio yet".to_string();
            return;
        }

        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("response")
            .to_string();

        let loaded = match ir::load_impulse_response(path, sample_rate as f64) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.ir_status = format!("{name}: {error}");
                return;
            }
        };

        if !self.ir_slot.publish(&loaded.taps) {
            self.ir_status = "another response is still being loaded".to_string();
            return;
        }

        self.remember_path(&path.to_string_lossy());
        self.ir_cabinet = cabinet_label(&path.to_string_lossy());
        self.ir_status = describe(&loaded, &name, sample_rate);
        self.browser_open = false;
    }

    /// Goes back to the synthesised 4x12.
    pub fn use_default_cabinet(&mut self) {
        if !self.ir_slot.publish_default() {
            self.ir_status = "another response is still being loaded".to_string();
            return;
        }
        self.remember_path("");
        self.ir_cabinet = DEFAULT_CABINET_LABEL.to_string();
        self.ir_status = "using the built-in 4x12".to_string();
        self.browser_open = false;
    }

    /// Opens or closes the browser, refreshing the listing on the way in.
    pub fn toggle_browser(&mut self) {
        self.browser_open = !self.browser_open;
        if self.browser_open {
            let start = self.ir_location.clone();
            self.browse_to(start);
        }
    }

    /// Records the loaded path in the persisted parameter.
    ///
    /// A poisoned lock means another thread panicked while holding it. The
    /// response itself has already been published and is audible; only the
    /// memory of it across a session reload is lost, so this reports rather
    /// than aborting the load.
    fn remember_path(&mut self, path: &str) {
        match self.params.ir_path.write() {
            Ok(mut stored) => {
                stored.clear();
                stored.push_str(path);
            }
            Err(_) => {
                self.ir_status =
                    "loaded, but this cabinet will not be saved with the project".to_string();
            }
        }
    }
}

/// One-line summary of what was loaded, for the status strip.
fn describe(loaded: &ir::LoadedIr, name: &str, sample_rate: f32) -> String {
    let mut summary = format!("{name} — {} taps", loaded.taps.len());
    if loaded.source_rate as f32 != sample_rate {
        summary.push_str(&format!(
            ", resampled {} → {} Hz",
            loaded.source_rate, sample_rate as u32
        ));
    }
    if loaded.source_channels > 1 {
        summary.push_str(&format!(", channel 1 of {}", loaded.source_channels));
    }
    if loaded.truncated {
        summary.push_str(", tail trimmed");
    }
    summary
}

/// The launcher button, which sits under the brand block and names the cabinet
/// currently loaded.
pub fn cabinet_button(cx: &mut Context) {
    Button::new(
        cx,
        |context| context.emit(UiEvent::ToggleBrowser),
        |cx| Label::new(cx, EditorData::ir_cabinet),
    )
    .class("cab-button");
}

/// The browser overlay. Present in the tree at all times; the binding builds
/// its contents only while it is open, so a closed browser costs nothing.
pub fn browser_overlay(cx: &mut Context) {
    Binding::new(cx, EditorData::browser_open, |cx, open| {
        if !open.get(cx) {
            return;
        }

        // A full-bleed scrim, which also swallows clicks aimed at the knobs
        // behind the panel.
        Element::new(cx).class("ir-scrim").on_press(|context| {
            context.emit(UiEvent::ToggleBrowser);
        });

        VStack::new(cx, |cx| {
            Label::new(cx, "CABINET IMPULSE RESPONSE").class("ir-title");
            Label::new(cx, EditorData::ir_directory).class("ir-path");

            HStack::new(cx, |cx| {
                Button::new(
                    cx,
                    |context| context.emit(UiEvent::BrowseUp),
                    |cx| Label::new(cx, "◄ UP"),
                )
                .class("ir-action");
                Button::new(
                    cx,
                    |context| context.emit(UiEvent::UseDefaultCabinet),
                    |cx| Label::new(cx, "BUILT-IN 4X12"),
                )
                .class("ir-action");
                Button::new(
                    cx,
                    |context| context.emit(UiEvent::ToggleBrowser),
                    |cx| Label::new(cx, "CLOSE"),
                )
                .class("ir-action");
            })
            .class("ir-toolbar");

            ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                VStack::new(cx, |cx| {
                    Binding::new(cx, EditorData::ir_entries, |cx, entries| {
                        // Collected first: the loop needs `cx` mutably.
                        let rows = entries.get(cx);
                        for (index, entry) in rows.iter().enumerate() {
                            let caption = if entry.is_directory {
                                format!("▸  {}", entry.label)
                            } else {
                                format!("     {}", entry.label)
                            };
                            Button::new(
                                cx,
                                move |context| context.emit(UiEvent::SelectEntry(index)),
                                |cx| Label::new(cx, &caption),
                            )
                            .class(if entry.is_directory {
                                "ir-folder"
                            } else {
                                "ir-file"
                            });
                        }
                    });
                })
                .class("ir-rows");
            })
            .class("ir-list");

            Label::new(cx, EditorData::ir_status).class("ir-status");
        })
        .class("ir-browser");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_cabinet_has_a_label_and_short_names_pass_through() {
        assert_eq!(cabinet_label(""), DEFAULT_CABINET_LABEL);
        assert_eq!(cabinet_label("C:/cabs/v30.wav"), "V30");
        assert_eq!(cabinet_label("/home/user/cabs/Green 25.wav"), "GREEN 25");
    }

    #[test]
    fn long_cabinet_names_keep_the_part_that_distinguishes_them() {
        // Two files from the same vendor pack differ only at the end. Eliding
        // from the front is what keeps them apart on the button.
        let first = short_cabinet_label("Marshall1960A_V30_SM57_Cone");
        let second = short_cabinet_label("Marshall1960A_V30_SM57_Cap");
        assert_ne!(first, second);
        assert!(first.chars().count() <= MAX_CABINET_LABEL);
        assert!(second.chars().count() <= MAX_CABINET_LABEL);
        assert!(first.ends_with("CONE"), "{first}");
        assert!(second.ends_with("CAP"), "{second}");

        // A name that already fits is left alone apart from the case.
        assert_eq!(short_cabinet_label("v30 cone"), "V30 CONE");
    }

    #[test]
    fn label_shortening_counts_characters_not_bytes() {
        // Slicing by byte would panic in the middle of these.
        let label = short_cabinet_label("åäöéüñ-cabinet-impulse-µ");
        assert!(label.chars().count() <= MAX_CABINET_LABEL);
        assert!(!label.is_empty());
    }

    #[test]
    fn directories_render_with_forward_slashes_and_bounded_length() {
        let short = display_directory(Path::new("C:\\cabs\\v30"));
        assert_eq!(short, "C:/cabs/v30");

        let deep = Path::new(
            "C:\\a-very-long-library-root\\vendor\\pack\\model\\mic\\position\\distance\\take",
        );
        let rendered = display_directory(deep);
        assert!(rendered.chars().count() <= 58, "{rendered}");
        assert!(rendered.starts_with('…'));
        assert!(
            rendered.ends_with("take"),
            "the leaf folder was elided away"
        );
    }

    #[test]
    fn the_status_line_reports_what_was_actually_done() {
        let plain = ir::LoadedIr {
            taps: vec![1.0; 512],
            source_rate: 48_000,
            source_channels: 1,
            source_frames: 512,
            truncated: false,
        };
        let summary = describe(&plain, "v30.wav", 48_000.0);
        assert_eq!(summary, "v30.wav — 512 taps");

        let converted = ir::LoadedIr {
            taps: vec![1.0; 4096],
            source_rate: 44_100,
            source_channels: 2,
            source_frames: 9_000,
            truncated: true,
        };
        let summary = describe(&converted, "long.wav", 96_000.0);
        assert!(summary.contains("resampled 44100 → 96000 Hz"), "{summary}");
        assert!(summary.contains("channel 1 of 2"), "{summary}");
        assert!(summary.contains("tail trimmed"), "{summary}");
    }
}
