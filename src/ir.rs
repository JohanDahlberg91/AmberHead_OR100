//! Loading a measured impulse response from a WAV file.
//!
//! Everything in this module runs on the editor thread or in
//! `Plugin::initialize`, never on the audio thread: it reads files, allocates,
//! and formats error strings. The result is a plain `Vec<f32>` of taps that
//! [`crate::shared::IrSlot`] carries across to the audio thread without either
//! side locking.
//!
//! # Why the decoder is written out longhand
//!
//! `CLAUDE.md` requires explicit approval before a crate is added, and the
//! project is approved for `nih_plug`, `nih_plug_vizia`, `realfft` and `wide`
//! only. A cabinet IR is a mono or stereo WAV of a few thousand frames in one
//! of five sample formats, which is a small enough target to decode directly —
//! and doing so keeps every bounds check and every rejected file visible here
//! rather than behind a dependency.
//!
//! # What is supported
//!
//! * `RIFF`/`WAVE` containers with the chunks in any order and with unknown
//!   chunks (`LIST`, `fact`, `cue `, ...) skipped.
//! * `WAVE_FORMAT_PCM` at 8, 16, 24 and 32 bits, `WAVE_FORMAT_IEEE_FLOAT` at
//!   32 and 64 bits, and `WAVE_FORMAT_EXTENSIBLE` wrapping either.
//! * Any channel count; **channel 0 is used** and the rest are discarded. A
//!   true-stereo cabinet capture would need two independent convolvers, and
//!   this amplifier is mono from the input jack onwards.
//! * Any sample rate, resampled to the host's rate by [`resample`].

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::dsp::cabinet::IR_LENGTH;
use crate::dsp::oversampling::{bessel_i0, sinc};

/// Longest impulse response the convolver can hold, in taps.
pub const MAX_IR_TAPS: usize = IR_LENGTH;

/// Largest file the loader will read, in bytes.
///
/// A 200 ms stereo 32-bit float IR at 96 kHz is 154 kB. 64 MB is four hundred
/// times that, which admits any plausible cabinet capture while refusing to
/// pull a multi-gigabyte file into memory because a user clicked the wrong row.
pub const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Half-width of the resampling kernel, in source samples at unity ratio.
///
/// 24 taps either side of the fractional position, Kaiser-windowed at
/// `beta = 8.6`, puts the resampler's stopband below -90 dB — the same
/// specification the oversampling cascade's first stage is designed to, so
/// converting a 44.1 kHz IR to 96 kHz adds nothing measurable to the noise
/// floor the rest of the chain already has.
const RESAMPLE_HALF_WIDTH: usize = 24;
/// Kaiser `beta` for that kernel. `beta = 0.1102 * (A - 8.7)` for `A = 90 dB`.
const RESAMPLE_BETA: f64 = 8.94;

/// Taps over which a truncated response is faded out.
///
/// An IR longer than [`MAX_IR_TAPS`] has to be cut somewhere, and cutting it
/// mid-tail leaves a step discontinuity that convolves into an audible click on
/// every transient. 128 taps is 2.7 ms at 48 kHz: long enough to be inaudible,
/// short enough not to shorten the usable tail.
const TRUNCATION_FADE_TAPS: usize = 128;

/// Highest sample rate accepted in a file header, in Hz. Anything above this is
/// a misparsed header rather than a real recording.
const MAX_SOURCE_RATE: u32 = 768_000;

/// `WAVE_FORMAT_PCM`.
const FORMAT_PCM: u16 = 1;
/// `WAVE_FORMAT_IEEE_FLOAT`.
const FORMAT_IEEE_FLOAT: u16 = 3;
/// `WAVE_FORMAT_EXTENSIBLE`; the real format is the first field of its
/// sub-format GUID, at offset 24 of the `fmt ` chunk.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Everything that can stop an impulse response from being loaded.
///
/// Each variant carries what the user needs to fix the problem, because the
/// editor shows this text verbatim and has nowhere else to look.
///
/// Not `Eq`: [`Self::BadTargetRate`] carries the offending rate, and `f64` has
/// no total equality.
#[derive(Debug, Clone, PartialEq)]
pub enum IrError {
    /// The file could not be opened or read.
    Io(String),
    /// The file is larger than [`MAX_FILE_BYTES`].
    TooLarge {
        /// Size reported by the filesystem, in bytes.
        bytes: u64,
    },
    /// The first twelve bytes are not `RIFF....WAVE`.
    NotRiffWave,
    /// A chunk header ran off the end of the file.
    Truncated,
    /// No `fmt ` chunk, or one shorter than the mandatory sixteen bytes.
    MissingFormat,
    /// No `data` chunk.
    MissingData,
    /// A format tag or bit depth this decoder does not handle.
    UnsupportedFormat {
        /// The `wFormatTag` field, resolved through `WAVE_FORMAT_EXTENSIBLE`.
        code: u16,
        /// The `wBitsPerSample` field.
        bits: u16,
    },
    /// The header declares zero channels or a zero bit depth.
    EmptyFormat,
    /// The header's sample rate is zero or beyond `MAX_SOURCE_RATE`.
    BadSampleRate(u32),
    /// The `data` chunk holds no complete frames.
    NoFrames,
    /// The decoded samples contain a NaN or an infinity.
    NonFinite,
    /// The decoded response is silent, so there is nothing to convolve with.
    Silent,
    /// The host sample rate handed to the loader is not a usable rate.
    BadTargetRate(f64),
}

impl fmt::Display for IrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "cannot read file: {message}"),
            Self::TooLarge { bytes } => write!(
                formatter,
                "file is {} MB, limit is {} MB",
                bytes / (1024 * 1024),
                MAX_FILE_BYTES / (1024 * 1024)
            ),
            Self::NotRiffWave => write!(formatter, "not a RIFF/WAVE file"),
            Self::Truncated => write!(formatter, "file is truncated"),
            Self::MissingFormat => write!(formatter, "no usable 'fmt ' chunk"),
            Self::MissingData => write!(formatter, "no 'data' chunk"),
            Self::UnsupportedFormat { code, bits } => write!(
                formatter,
                "unsupported format {code} at {bits}-bit; \
                 use 16/24/32-bit PCM or 32/64-bit float"
            ),
            Self::EmptyFormat => write!(formatter, "header declares no channels or no bits"),
            Self::BadSampleRate(rate) => write!(formatter, "implausible sample rate {rate} Hz"),
            Self::NoFrames => write!(formatter, "no audio frames"),
            Self::NonFinite => write!(formatter, "samples contain NaN or infinity"),
            Self::Silent => write!(formatter, "response is silent"),
            Self::BadTargetRate(rate) => write!(formatter, "invalid host sample rate {rate}"),
        }
    }
}

impl std::error::Error for IrError {}

/// One decoded WAV file, reduced to its first channel.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedWav {
    /// Sample rate declared by the `fmt ` chunk.
    pub sample_rate: u32,
    /// Channel count declared by the `fmt ` chunk, before reduction.
    pub channels: u16,
    /// Channel 0, as normalized floats.
    pub samples: Vec<f32>,
}

/// An impulse response ready to hand to the convolver.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedIr {
    /// At most [`MAX_IR_TAPS`] taps, at the host sample rate.
    pub taps: Vec<f32>,
    /// Rate the file was recorded at, before resampling.
    pub source_rate: u32,
    /// Channel count of the file, before channel 0 was taken.
    pub source_channels: u16,
    /// Frames the file held, before resampling and truncation.
    pub source_frames: usize,
    /// Whether the response had to be cut to [`MAX_IR_TAPS`].
    pub truncated: bool,
}

/// Reads a little-endian `u16` at `offset`, or `None` past the end.
fn u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    let slice = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([*slice.first()?, *slice.get(1)?]))
}

/// Reads a little-endian `u32` at `offset`, or `None` past the end.
fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([
        *slice.first()?,
        *slice.get(1)?,
        *slice.get(2)?,
        *slice.get(3)?,
    ]))
}

/// Decodes one PCM or float sample of `bits` width starting at `offset`.
///
/// Returns `None` only when the slice is short, which the frame loop has
/// already ruled out; the redundant check is what keeps this function free of
/// panicking indexing (`CLAUDE.md` §1).
fn sample_at(bytes: &[u8], offset: usize, format: u16, bits: u16) -> Option<f32> {
    match (format, bits) {
        // 8-bit PCM is unsigned with a 128 offset, unlike every wider depth.
        (FORMAT_PCM, 8) => {
            let raw = *bytes.get(offset)? as i16 - 128;
            Some(raw as f32 / 128.0)
        }
        (FORMAT_PCM, 16) => {
            let raw = u16_at(bytes, offset)? as i16;
            Some(raw as f32 / 32_768.0)
        }
        (FORMAT_PCM, 24) => {
            let low = *bytes.get(offset)? as i32;
            let mid = *bytes.get(offset + 1)? as i32;
            let high = *bytes.get(offset + 2)? as i32;
            // Sign-extend the 24-bit value by shifting it into the top of an
            // i32 and back down again.
            let raw = ((low | (mid << 8) | (high << 16)) << 8) >> 8;
            Some(raw as f32 / 8_388_608.0)
        }
        (FORMAT_PCM, 32) => {
            let raw = u32_at(bytes, offset)? as i32;
            Some(raw as f32 / 2_147_483_648.0)
        }
        (FORMAT_IEEE_FLOAT, 32) => Some(f32::from_bits(u32_at(bytes, offset)?)),
        (FORMAT_IEEE_FLOAT, 64) => {
            let low = u32_at(bytes, offset)? as u64;
            let high = u32_at(bytes, offset + 4)? as u64;
            Some(f64::from_bits(low | (high << 32)) as f32)
        }
        _ => None,
    }
}

/// Parses a `RIFF`/`WAVE` byte stream into channel 0 of its audio.
///
/// Chunks may appear in any order and unknown chunks are skipped, which is what
/// lets this read the `LIST`-prefixed files most IR vendors ship.
pub fn decode_wav(bytes: &[u8]) -> Result<DecodedWav, IrError> {
    if bytes.len() < 12 {
        return Err(IrError::Truncated);
    }
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err(IrError::NotRiffWave);
    }

    let mut format_chunk: Option<&[u8]> = None;
    let mut data_chunk: Option<&[u8]> = None;

    let mut cursor = 12usize;
    // Every iteration consumes at least eight bytes, so this terminates.
    while cursor + 8 <= bytes.len() {
        let Some(id) = bytes.get(cursor..cursor + 4) else {
            return Err(IrError::Truncated);
        };
        let Some(size) = u32_at(bytes, cursor + 4) else {
            return Err(IrError::Truncated);
        };
        let body_start = cursor + 8;
        // A declared size past the end of the file is tolerated by clamping,
        // because a recorder killed mid-write leaves exactly that and the
        // audio before the cut is still perfectly good.
        let body_end = body_start.saturating_add(size as usize).min(bytes.len());
        let Some(body) = bytes.get(body_start..body_end) else {
            return Err(IrError::Truncated);
        };

        match id {
            b"fmt " => format_chunk = Some(body),
            b"data" => data_chunk = Some(body),
            _ => {}
        }

        // Chunks are padded to an even length; the pad byte is not counted in
        // the declared size.
        let advance = (size as usize).saturating_add(size as usize & 1);
        cursor = body_start.saturating_add(advance);
    }

    let Some(format_chunk) = format_chunk else {
        return Err(IrError::MissingFormat);
    };
    if format_chunk.len() < 16 {
        return Err(IrError::MissingFormat);
    }
    let Some(data) = data_chunk else {
        return Err(IrError::MissingData);
    };

    let (Some(mut format), Some(channels), Some(sample_rate), Some(bits)) = (
        u16_at(format_chunk, 0),
        u16_at(format_chunk, 2),
        u32_at(format_chunk, 4),
        u16_at(format_chunk, 14),
    ) else {
        return Err(IrError::MissingFormat);
    };

    if format == FORMAT_EXTENSIBLE {
        // The sub-format GUID's first two bytes hold the real format tag.
        let Some(sub_format) = u16_at(format_chunk, 24) else {
            return Err(IrError::MissingFormat);
        };
        format = sub_format;
    }

    if channels == 0 || bits == 0 || !bits.is_multiple_of(8) {
        return Err(IrError::EmptyFormat);
    }
    if sample_rate == 0 || sample_rate > MAX_SOURCE_RATE {
        return Err(IrError::BadSampleRate(sample_rate));
    }
    if sample_at(&[0u8; 8], 0, format, bits).is_none() {
        return Err(IrError::UnsupportedFormat { code: format, bits });
    }

    let sample_bytes = (bits / 8) as usize;
    let frame_bytes = sample_bytes.saturating_mul(channels as usize);
    if frame_bytes == 0 {
        return Err(IrError::EmptyFormat);
    }
    let frames = data.len() / frame_bytes;
    if frames == 0 {
        return Err(IrError::NoFrames);
    }

    let mut samples = Vec::with_capacity(frames);
    for frame in 0..frames {
        // Channel 0 is the first sample of the frame.
        let Some(value) = sample_at(data, frame * frame_bytes, format, bits) else {
            return Err(IrError::Truncated);
        };
        if !value.is_finite() {
            return Err(IrError::NonFinite);
        }
        samples.push(value);
    }

    Ok(DecodedWav {
        sample_rate,
        channels,
        samples,
    })
}

/// Kaiser window value at `position`, normalized so `|position| <= 1`.
fn kaiser(position: f64, i0_beta: f64) -> f64 {
    let squared = (1.0 - position * position).max(0.0);
    bessel_i0(RESAMPLE_BETA * squared.sqrt()) / i0_beta
}

/// Converts `input` from `from_rate` to `to_rate` by windowed-sinc
/// interpolation.
///
/// The kernel is `2*fc * sinc(2*fc*u)` under a Kaiser window, with `fc` at half
/// the *lower* of the two rates so that downsampling band-limits before it
/// decimates. Its integer sum is unity, so the conversion preserves DC gain and
/// therefore the response's level.
///
/// Returns an empty vector for an empty input or a non-positive rate; returns
/// the input unchanged when the rates already match, which is the common case
/// for an IR recorded at the session rate.
pub fn resample(input: &[f32], from_rate: f64, to_rate: f64) -> Vec<f32> {
    let rates_usable =
        from_rate.is_finite() && from_rate > 0.0 && to_rate.is_finite() && to_rate > 0.0;
    if input.is_empty() || !rates_usable {
        return Vec::new();
    }
    if (from_rate - to_rate).abs() < 1.0e-6 {
        return input.to_vec();
    }

    let ratio = to_rate / from_rate;
    // Cutoff in cycles per *input* sample: 0.5 when upsampling, tightened to
    // the output's Nyquist when downsampling.
    let cutoff = 0.5 * ratio.min(1.0);
    // Widening the kernel by the same factor keeps the transition band as
    // sharp in absolute terms as it is at unity ratio.
    let half_width = RESAMPLE_HALF_WIDTH as f64 * 0.5 / cutoff;
    let i0_beta = bessel_i0(RESAMPLE_BETA);

    let output_len = ((input.len() as f64) * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        // Position of this output sample expressed in input samples.
        let centre = index as f64 / ratio;
        let first = (centre - half_width).ceil().max(0.0) as usize;
        let last = ((centre + half_width).floor().max(0.0) as usize).min(input.len() - 1);

        let mut accumulator = 0.0f64;
        for source in first..=last {
            let offset = centre - source as f64;
            let window = kaiser(offset / half_width, i0_beta);
            let tap = 2.0 * cutoff * sinc(2.0 * cutoff * offset) * window;
            accumulator += input.get(source).copied().unwrap_or(0.0) as f64 * tap;
        }
        output.push(accumulator as f32);
    }
    output
}

/// Trims `samples` to [`MAX_IR_TAPS`], fading out across the cut.
///
/// Returns the taps and whether a cut was made.
fn truncate_with_fade(mut samples: Vec<f32>) -> (Vec<f32>, bool) {
    if samples.len() <= MAX_IR_TAPS {
        return (samples, false);
    }
    samples.truncate(MAX_IR_TAPS);

    let fade = TRUNCATION_FADE_TAPS.min(MAX_IR_TAPS);
    let start = MAX_IR_TAPS - fade;
    for (step, tap) in samples.iter_mut().skip(start).enumerate() {
        // Raised cosine from 1 down to 0 across the fade, so both the value and
        // its slope reach zero at the cut.
        let position = (step + 1) as f32 / fade as f32;
        let gain = 0.5 + 0.5 * (std::f32::consts::PI * position).cos();
        *tap *= gain;
    }
    (samples, true)
}

/// Converts decoded audio into convolver taps at `target_rate`.
///
/// Resamples if needed, cuts to [`MAX_IR_TAPS`] with a fade, and rejects a
/// response that came out silent or non-finite. The result is *not* normalised
/// here: levelling belongs to
/// [`crate::dsp::cabinet::normalise_to_reference_band`], which is applied to
/// the loaded and the synthesised response alike so the two match.
pub fn prepare_taps(decoded: &DecodedWav, target_rate: f64) -> Result<LoadedIr, IrError> {
    if !target_rate.is_finite() || target_rate <= 0.0 || target_rate > MAX_SOURCE_RATE as f64 {
        return Err(IrError::BadTargetRate(target_rate));
    }
    if decoded.samples.is_empty() {
        return Err(IrError::NoFrames);
    }

    let converted = resample(&decoded.samples, decoded.sample_rate as f64, target_rate);
    let (taps, truncated) = truncate_with_fade(converted);

    if taps.is_empty() {
        return Err(IrError::NoFrames);
    }
    if taps.iter().any(|tap| !tap.is_finite()) {
        return Err(IrError::NonFinite);
    }
    if taps.iter().all(|tap| *tap == 0.0) {
        return Err(IrError::Silent);
    }

    Ok(LoadedIr {
        taps,
        source_rate: decoded.sample_rate,
        source_channels: decoded.channels,
        source_frames: decoded.samples.len(),
        truncated,
    })
}

/// Reads a WAV file and converts it into convolver taps at `target_rate`.
///
/// Blocking file I/O: call from the editor thread or from
/// `Plugin::initialize`, never from `Plugin::process`.
pub fn load_impulse_response(path: &Path, target_rate: f64) -> Result<LoadedIr, IrError> {
    let metadata = fs::metadata(path).map_err(|error| IrError::Io(error.to_string()))?;
    if metadata.len() > MAX_FILE_BYTES {
        return Err(IrError::TooLarge {
            bytes: metadata.len(),
        });
    }
    let bytes = fs::read(path).map_err(|error| IrError::Io(error.to_string()))?;
    let decoded = decode_wav(&bytes)?;
    prepare_taps(&decoded, target_rate)
}

/// One row of the editor's file browser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryEntry {
    /// Text shown in the list.
    pub label: String,
    /// Where selecting the row leads.
    pub path: PathBuf,
    /// Whether selecting it navigates rather than loads.
    pub is_directory: bool,
}

/// Extensions the browser offers as loadable.
const IR_EXTENSIONS: [&str; 2] = ["wav", "wave"];

/// Lists `directory`, keeping sub-directories and WAV files.
///
/// Directories sort before files and both sort case-insensitively, so the list
/// does not reorder itself between platforms. Unreadable entries are skipped
/// rather than failing the whole listing: one permission-denied folder in a
/// sample library should not make the library unbrowsable.
pub fn list_directory(directory: &Path) -> Result<Vec<DirectoryEntry>, IrError> {
    let reader = fs::read_dir(directory).map_err(|error| IrError::Io(error.to_string()))?;

    let mut directories = Vec::new();
    let mut files = Vec::new();
    for entry in reader.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        // Hidden entries are noise in a sample library and, on Unix, are most
        // of what a home directory contains.
        if name.starts_with('.') {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_dir() {
            directories.push(DirectoryEntry {
                label: name.to_string(),
                path,
                is_directory: true,
            });
        } else if has_ir_extension(&path) {
            files.push(DirectoryEntry {
                label: name.to_string(),
                path,
                is_directory: false,
            });
        }
    }

    let by_label = |left: &DirectoryEntry, right: &DirectoryEntry| {
        left.label
            .to_lowercase()
            .cmp(&right.label.to_lowercase())
            .then_with(|| left.label.cmp(&right.label))
    };
    directories.sort_by(by_label);
    files.sort_by(by_label);
    directories.append(&mut files);
    Ok(directories)
}

/// Whether `path` ends in an extension the browser offers.
pub fn has_ir_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let lowered = extension.to_lowercase();
            IR_EXTENSIONS.contains(&lowered.as_str())
        })
        .unwrap_or(false)
}

/// Directory the browser should open at.
///
/// Prefers the folder of the response already loaded, so reopening the browser
/// lands where the user last was; otherwise the user's home directory, and
/// finally the process's working directory, which always exists.
pub fn starting_directory(current: &str) -> PathBuf {
    if !current.is_empty() {
        let path = Path::new(current);
        if let Some(parent) = path.parent() {
            if parent.is_dir() {
                return parent.to_path_buf();
            }
        }
    }
    for variable in ["USERPROFILE", "HOME"] {
        if let Ok(home) = std::env::var(variable) {
            let path = PathBuf::from(home);
            if path.is_dir() {
                return path;
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `RIFF`/`WAVE` byte stream around `data`.
    fn wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&format.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * block_align as u32).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());

        let mut body = Vec::new();
        body.extend_from_slice(b"WAVE");
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(data);

        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);
        file
    }

    fn f32_data(samples: &[f32]) -> Vec<u8> {
        samples
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn decodes_16_bit_pcm() {
        let data: Vec<u8> = [0i16, 16_384, -16_384, i16::MIN]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let decoded = decode_wav(&wav(FORMAT_PCM, 1, 48_000, 16, &data)).expect("decode failed");
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples, vec![0.0, 0.5, -0.5, -1.0]);
    }

    #[test]
    fn decodes_24_bit_pcm_with_correct_sign_extension() {
        // 0x000000 = 0, 0x400000 = +0.5, 0xC00000 = -0.5, 0x800000 = -1.0.
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x00, // 0
            0x00, 0x00, 0x40, // +0.5
            0x00, 0x00, 0xC0, // -0.5
            0x00, 0x00, 0x80, // -1.0
        ];
        let decoded = decode_wav(&wav(FORMAT_PCM, 1, 44_100, 24, &data)).expect("decode failed");
        assert_eq!(decoded.samples, vec![0.0, 0.5, -0.5, -1.0]);
    }

    #[test]
    fn decodes_8_and_32_bit_pcm() {
        let eight = decode_wav(&wav(FORMAT_PCM, 1, 48_000, 8, &[128, 192, 64, 0]))
            .expect("8-bit decode failed");
        assert_eq!(eight.samples, vec![0.0, 0.5, -0.5, -1.0]);

        let data: Vec<u8> = [0i32, 1_073_741_824, -1_073_741_824]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let thirty_two =
            decode_wav(&wav(FORMAT_PCM, 1, 48_000, 32, &data)).expect("32-bit decode failed");
        assert_eq!(thirty_two.samples, vec![0.0, 0.5, -0.5]);
    }

    #[test]
    fn decodes_32_and_64_bit_float() {
        let single = decode_wav(&wav(
            FORMAT_IEEE_FLOAT,
            1,
            96_000,
            32,
            &f32_data(&[0.25, -0.75, 1.0]),
        ))
        .expect("f32 decode failed");
        assert_eq!(single.samples, vec![0.25, -0.75, 1.0]);

        let data: Vec<u8> = [0.25f64, -0.75, 1.0]
            .iter()
            .flat_map(|sample| sample.to_bits().to_le_bytes())
            .collect();
        let double =
            decode_wav(&wav(FORMAT_IEEE_FLOAT, 1, 96_000, 64, &data)).expect("f64 decode failed");
        assert_eq!(double.samples, vec![0.25, -0.75, 1.0]);
    }

    #[test]
    fn takes_channel_zero_of_a_multichannel_file() {
        // Interleaved stereo: left ramps up, right is held at -1.
        let samples: Vec<f32> = (0..4).flat_map(|n| [n as f32 * 0.25, -1.0]).collect();
        let decoded = decode_wav(&wav(FORMAT_IEEE_FLOAT, 2, 48_000, 32, &f32_data(&samples)))
            .expect("decode failed");
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples, vec![0.0, 0.25, 0.5, 0.75]);
    }

    #[test]
    fn resolves_wave_format_extensible_to_its_sub_format() {
        // A 40-byte `fmt ` chunk: the 16 mandatory bytes, cbSize, the union,
        // the channel mask, then a 16-byte GUID whose first field is the real
        // format tag.
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes()); // channels
        fmt.extend_from_slice(&48_000u32.to_le_bytes());
        fmt.extend_from_slice(&(48_000u32 * 4).to_le_bytes());
        fmt.extend_from_slice(&4u16.to_le_bytes()); // block align
        fmt.extend_from_slice(&32u16.to_le_bytes()); // bits
        fmt.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        fmt.extend_from_slice(&32u16.to_le_bytes()); // valid bits
        fmt.extend_from_slice(&4u32.to_le_bytes()); // channel mask
        fmt.extend_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
        fmt.extend_from_slice(&[0u8; 14]); // rest of the GUID

        let data = f32_data(&[0.5, -0.5]);
        let mut body = Vec::new();
        body.extend_from_slice(b"WAVE");
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(&data);

        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);

        let decoded = decode_wav(&file).expect("extensible decode failed");
        assert_eq!(decoded.samples, vec![0.5, -0.5]);
    }

    #[test]
    fn skips_unknown_chunks_in_any_order() {
        // `LIST` before `fmt `, an odd-sized chunk needing its pad byte, and
        // `data` last — the layout most IR vendors actually ship.
        let data = f32_data(&[1.0, 0.5]);
        let mut body = Vec::new();
        body.extend_from_slice(b"WAVE");
        body.extend_from_slice(b"LIST");
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(b"INFO\0");
        body.push(0); // pad to even

        let mut fmt = Vec::new();
        fmt.extend_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&48_000u32.to_le_bytes());
        fmt.extend_from_slice(&(48_000u32 * 4).to_le_bytes());
        fmt.extend_from_slice(&4u16.to_le_bytes());
        fmt.extend_from_slice(&32u16.to_le_bytes());
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);

        body.extend_from_slice(b"data");
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(&data);

        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);

        assert_eq!(
            decode_wav(&file).expect("decode failed").samples,
            vec![1.0, 0.5]
        );
    }

    #[test]
    fn rejects_every_malformed_header_with_a_specific_error() {
        assert_eq!(decode_wav(&[]), Err(IrError::Truncated));
        assert_eq!(
            decode_wav(b"not a wave at all!!"),
            Err(IrError::NotRiffWave)
        );

        // RIFF but not WAVE.
        let mut avi = b"RIFF".to_vec();
        avi.extend_from_slice(&8u32.to_le_bytes());
        avi.extend_from_slice(b"AVI JUNK");
        assert_eq!(decode_wav(&avi), Err(IrError::NotRiffWave));

        // WAVE with no chunks at all.
        let mut bare = b"RIFF".to_vec();
        bare.extend_from_slice(&4u32.to_le_bytes());
        bare.extend_from_slice(b"WAVE");
        assert_eq!(decode_wav(&bare), Err(IrError::MissingFormat));

        // A format nothing decodes: 12-bit is not a multiple this reads.
        assert_eq!(
            decode_wav(&wav(FORMAT_PCM, 1, 48_000, 64, &[0u8; 16])),
            Err(IrError::UnsupportedFormat {
                code: FORMAT_PCM,
                bits: 64
            })
        );
        assert!(matches!(
            decode_wav(&wav(999, 1, 48_000, 16, &[0u8; 4])),
            Err(IrError::UnsupportedFormat { code: 999, .. })
        ));

        assert_eq!(
            decode_wav(&wav(FORMAT_PCM, 0, 48_000, 16, &[0u8; 4])),
            Err(IrError::EmptyFormat)
        );
        assert_eq!(
            decode_wav(&wav(FORMAT_PCM, 1, 0, 16, &[0u8; 4])),
            Err(IrError::BadSampleRate(0))
        );
        assert_eq!(
            decode_wav(&wav(FORMAT_PCM, 1, 48_000, 16, &[])),
            Err(IrError::NoFrames)
        );
        assert_eq!(
            decode_wav(&wav(
                FORMAT_IEEE_FLOAT,
                1,
                48_000,
                32,
                &f32_data(&[f32::NAN])
            )),
            Err(IrError::NonFinite)
        );
    }

    #[test]
    fn a_truncated_data_chunk_yields_the_frames_that_survived() {
        // Declare eight frames, supply three. A recorder killed mid-write
        // leaves exactly this, and the audio before the cut is still good.
        let mut file = wav(
            FORMAT_IEEE_FLOAT,
            1,
            48_000,
            32,
            &f32_data(&[1.0, 0.5, 0.25]),
        );
        let length = file.len();
        // Overwrite the `data` chunk size with a larger figure.
        let data_size_offset = length - 12 - 4;
        assert_eq!(
            file.get(data_size_offset - 4..data_size_offset),
            Some(&b"data"[..])
        );
        if let Some(field) = file.get_mut(data_size_offset..data_size_offset + 4) {
            field.copy_from_slice(&32u32.to_le_bytes());
        }
        assert_eq!(
            decode_wav(&file).expect("decode failed").samples,
            vec![1.0, 0.5, 0.25]
        );
    }

    #[test]
    fn resampling_at_a_matching_rate_is_the_identity() {
        let input: Vec<f32> = (0..64).map(|n| (n as f32 * 0.1).sin()).collect();
        assert_eq!(resample(&input, 48_000.0, 48_000.0), input);
        assert!(resample(&[], 48_000.0, 96_000.0).is_empty());
        assert!(resample(&input, 0.0, 48_000.0).is_empty());
        assert!(resample(&input, 48_000.0, 0.0).is_empty());
    }

    #[test]
    fn resampling_preserves_a_tone_and_its_level() {
        // A 1 kHz tone at 44.1 kHz, converted up to 48 kHz and down to
        // 32 kHz, must stay a 1 kHz tone at the same amplitude.
        let source_rate = 44_100.0f64;
        let hz = 1_000.0f64;
        let input: Vec<f32> = (0..8_192)
            .map(|n| (std::f64::consts::TAU * hz * n as f64 / source_rate).sin() as f32)
            .collect();

        for target_rate in [48_000.0f64, 96_000.0, 32_000.0] {
            let output = resample(&input, source_rate, target_rate);
            let expected_len = (input.len() as f64 * target_rate / source_rate).ceil() as usize;
            assert_eq!(output.len(), expected_len, "wrong length at {target_rate}");

            // Skip the kernel's edge transient at both ends.
            let margin = 256;
            let mut peak = 0.0f32;
            let mut worst = 0.0f32;
            for (index, sample) in output
                .iter()
                .enumerate()
                .skip(margin)
                .take(output.len().saturating_sub(2 * margin))
            {
                let reference =
                    (std::f64::consts::TAU * hz * index as f64 / target_rate).sin() as f32;
                peak = peak.max(sample.abs());
                worst = worst.max((sample - reference).abs());
            }
            assert!(
                (0.99..=1.01).contains(&peak),
                "level moved at {target_rate}: peak {peak}"
            );
            assert!(worst < 0.01, "waveform error {worst} at {target_rate}");
        }
    }

    #[test]
    fn downsampling_band_limits_before_it_decimates() {
        // A 15 kHz tone at 48 kHz has nowhere to go at 24 kHz. It must be
        // filtered away, not folded back to 9 kHz.
        let input: Vec<f32> = (0..8_192)
            .map(|n| (std::f64::consts::TAU * 15_000.0 * n as f64 / 48_000.0).sin() as f32)
            .collect();
        let output = resample(&input, 48_000.0, 24_000.0);
        let peak = output
            .iter()
            .skip(256)
            .take(output.len().saturating_sub(512))
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!(peak < 0.01, "aliased image survived at {peak}");
    }

    #[test]
    fn preparing_taps_truncates_long_responses_with_a_fade() {
        let decoded = DecodedWav {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![1.0; MAX_IR_TAPS * 2],
        };
        let loaded = prepare_taps(&decoded, 48_000.0).expect("prepare failed");
        assert_eq!(loaded.taps.len(), MAX_IR_TAPS);
        assert!(loaded.truncated);
        assert_eq!(loaded.source_frames, MAX_IR_TAPS * 2);

        // The response reaches exactly zero at the cut, and the tap before the
        // fade is untouched.
        assert_eq!(loaded.taps.last().copied(), Some(0.0));
        let before_fade = MAX_IR_TAPS - TRUNCATION_FADE_TAPS - 1;
        assert_eq!(loaded.taps.get(before_fade).copied(), Some(1.0));
        // And it is monotone across the fade, so no step is left anywhere.
        for index in before_fade..MAX_IR_TAPS - 1 {
            let (Some(current), Some(next)) = (
                loaded.taps.get(index).copied(),
                loaded.taps.get(index + 1).copied(),
            ) else {
                unreachable!("indices are inside a MAX_IR_TAPS-long vector")
            };
            assert!(next <= current + 1.0e-6, "fade rose at {index}");
        }
    }

    #[test]
    fn preparing_taps_keeps_short_responses_whole() {
        let decoded = DecodedWav {
            sample_rate: 48_000,
            channels: 2,
            samples: vec![1.0, 0.5, 0.25],
        };
        let loaded = prepare_taps(&decoded, 48_000.0).expect("prepare failed");
        assert_eq!(loaded.taps, vec![1.0, 0.5, 0.25]);
        assert!(!loaded.truncated);
        assert_eq!(loaded.source_channels, 2);
        assert_eq!(loaded.source_rate, 48_000);
    }

    #[test]
    fn preparing_taps_rejects_unusable_input() {
        let silent = DecodedWav {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![0.0; 64],
        };
        assert_eq!(prepare_taps(&silent, 48_000.0), Err(IrError::Silent));

        let empty = DecodedWav {
            sample_rate: 48_000,
            channels: 1,
            samples: Vec::new(),
        };
        assert_eq!(prepare_taps(&empty, 48_000.0), Err(IrError::NoFrames));

        let good = DecodedWav {
            sample_rate: 48_000,
            channels: 1,
            samples: vec![1.0, 0.5],
        };
        assert_eq!(prepare_taps(&good, 0.0), Err(IrError::BadTargetRate(0.0)));
        assert!(matches!(
            prepare_taps(&good, 1.0e9),
            Err(IrError::BadTargetRate(_))
        ));
    }

    #[test]
    fn every_error_renders_a_message_that_names_the_problem() {
        let errors = [
            IrError::Io("access denied".into()),
            IrError::TooLarge {
                bytes: 128 * 1024 * 1024,
            },
            IrError::NotRiffWave,
            IrError::Truncated,
            IrError::MissingFormat,
            IrError::MissingData,
            IrError::UnsupportedFormat { code: 2, bits: 4 },
            IrError::EmptyFormat,
            IrError::BadSampleRate(0),
            IrError::NoFrames,
            IrError::NonFinite,
            IrError::Silent,
            IrError::BadTargetRate(-1.0),
        ];
        for error in errors {
            let rendered = error.to_string();
            assert!(!rendered.is_empty(), "{error:?} rendered empty");
            assert!(
                rendered.chars().next().is_some_and(|c| !c.is_uppercase()),
                "{error:?} starts capitalised; it is shown mid-sentence"
            );
        }
        assert!(IrError::TooLarge {
            bytes: 128 * 1024 * 1024
        }
        .to_string()
        .contains("128 MB"));
    }

    #[test]
    fn extension_matching_is_case_insensitive_and_narrow() {
        assert!(has_ir_extension(Path::new("cab.wav")));
        assert!(has_ir_extension(Path::new("CAB.WAV")));
        assert!(has_ir_extension(Path::new("cab.Wave")));
        assert!(!has_ir_extension(Path::new("cab.aiff")));
        assert!(!has_ir_extension(Path::new("cab")));
        assert!(!has_ir_extension(Path::new("wav")));
    }

    #[test]
    fn the_starting_directory_always_exists() {
        // An empty or nonsense path must still land somewhere real, because
        // the browser has nothing to fall back to if it does not.
        assert!(starting_directory("").is_dir());
        assert!(starting_directory("/no/such/place/at/all/cab.wav").is_dir());

        // A real file's parent is preferred over the home directory.
        let here = std::env::current_dir().expect("no working directory");
        let file = here.join("Cargo.toml");
        assert_eq!(
            starting_directory(&file.to_string_lossy()),
            here,
            "the browser did not reopen where the response lives"
        );
    }

    #[test]
    fn listing_a_directory_finds_wavs_and_sorts_directories_first() {
        // The repository root is guaranteed to hold both directories and
        // non-WAV files, which is exactly the filtering this has to get right.
        let here = std::env::current_dir().expect("no working directory");
        let entries = list_directory(&here).expect("listing failed");

        assert!(
            entries.iter().any(|entry| entry.is_directory),
            "no directories found in the repository root"
        );
        assert!(
            entries
                .iter()
                .all(|entry| entry.is_directory || has_ir_extension(&entry.path)),
            "a non-WAV file was offered as loadable"
        );
        assert!(
            !entries.iter().any(|entry| entry.label == "Cargo.toml"),
            "Cargo.toml was listed"
        );
        assert!(
            !entries.iter().any(|entry| entry.label.starts_with('.')),
            "a hidden entry was listed"
        );

        // Directories first, then each group sorted.
        let split = entries
            .iter()
            .position(|entry| !entry.is_directory)
            .unwrap_or(entries.len());
        assert!(
            entries.iter().skip(split).all(|entry| !entry.is_directory),
            "directories and files are interleaved"
        );

        assert!(list_directory(Path::new("/no/such/place")).is_err());
    }

    #[test]
    fn a_wav_round_trips_from_disk_to_taps() {
        // The full path the editor takes, exercised against a real file.
        let mut path = std::env::temp_dir();
        path.push("amberhead_or100_ir_round_trip.wav");

        let samples: Vec<f32> = (0..512).map(|n| 0.9f32.powi(n) * 0.5).collect();
        let file = wav(FORMAT_IEEE_FLOAT, 1, 44_100, 32, &f32_data(&samples));
        fs::write(&path, &file).expect("could not write the test file");

        let loaded = load_impulse_response(&path, 48_000.0).expect("load failed");
        assert_eq!(loaded.source_rate, 44_100);
        assert_eq!(loaded.source_frames, 512);
        assert!(!loaded.truncated);
        // Resampled 44.1 -> 48 kHz, so slightly longer than the source.
        assert_eq!(
            loaded.taps.len(),
            (512.0 * 48_000.0 / 44_100.0f64).ceil() as usize
        );
        assert!(loaded.taps.iter().all(|tap| tap.is_finite()));
        assert!(loaded.taps.iter().any(|tap| tap.abs() > 0.1));

        let _ = fs::remove_file(&path);
        assert!(matches!(
            load_impulse_response(&path, 48_000.0),
            Err(IrError::Io(_))
        ));
    }
}
