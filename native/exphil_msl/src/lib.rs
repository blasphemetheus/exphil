//! NIF over melee-sim-light's C batch API (src/api.h): SIM_INTEGRATION.md
//! step 10b. The library is loaded at runtime (libloading) from the path the
//! caller gives, so the sim checkout is a data dependency, not a link one.
//!
//! Every row crosses the boundary as raw bytes in the C struct layout —
//! `MslMatchConfig` (52 B), `MslInput` (32 B), `MslObservation` (980 B),
//! `MslTerminal` (16 B) — which are byte-identical to the Python side's numpy
//! dtypes, so `ExPhil.Bridge.SimRows` decodes them unchanged.

use libloading::{Library, Symbol};
use rustler::{Binary, Env, NewBinary, ResourceArc};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Mutex;

const CONFIG_SIZE: usize = 52;
const INPUT_SIZE: usize = 32;
const OBS_SIZE: usize = 980;
const TERM_SIZE: usize = 16;

type CreateFn = unsafe extern "C" fn(*const c_char, u32, *mut *mut c_void) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type ResetFn = unsafe extern "C" fn(*mut c_void, *const u8, *const u8, *mut u8) -> c_int;
type StepFn = unsafe extern "C" fn(*mut c_void, *const u8, *mut u8, *mut u8) -> c_int;
type ObserveFn = unsafe extern "C" fn(*const c_void, *mut u8, *mut u8) -> c_int;
type SaveSizeFn = unsafe extern "C" fn(*const c_void, u32, *mut usize) -> c_int;
type SaveFn = unsafe extern "C" fn(*const c_void, u32, *mut u8, usize, *mut usize) -> c_int;
type RestoreFn = unsafe extern "C" fn(*mut c_void, u32, *const u8, usize) -> c_int;
type ResultStringFn = unsafe extern "C" fn(c_int) -> *const c_char;

#[repr(C)]
#[derive(Clone, Copy)]
struct MslPlayerConfig {
    character: u8,
    team: i8,
    facing: i8,
    controller_port: i8,
    costume: u8,
    handicap: u8,
    start_percent: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MslMatchConfig {
    stage: u32,
    random_seed: u32,
    max_frame: i32,
    damage_ratio: f32,
    num_players: u8,
    is_teams: u8,
    friendly_fire: u8,
    stocks: u8,
    viewpoint_player: u8,
    ucf_cardinals: u8,
    players: [MslPlayerConfig; 4],
}

type ConfigDefaultFn = unsafe extern "C" fn() -> MslMatchConfig;

struct Batch {
    lib: Library,
    ptr: *mut c_void,
    size: usize,
}

unsafe impl Send for Batch {}
unsafe impl Sync for Batch {}

pub struct BatchResource(Mutex<Batch>);

impl Drop for Batch {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                if let Ok(destroy) = self.lib.get::<DestroyFn>(b"msl_batch_destroy\0") {
                    destroy(self.ptr);
                }
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

fn sym<'a, T>(lib: &'a Library, name: &[u8]) -> Result<Symbol<'a, T>, String> {
    unsafe { lib.get::<T>(name).map_err(|e| format!("symbol {}: {}", String::from_utf8_lossy(name), e)) }
}

fn check(lib: &Library, code: c_int, op: &str) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe {
        match lib.get::<ResultStringFn>(b"msl_result_string\0") {
            Ok(f) => {
                let p = f(code);
                if p.is_null() { format!("code {}", code) } else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
            }
            Err(_) => format!("code {}", code),
        }
    };
    Err(format!("{}: {}", op, msg))
}

fn to_binary<'a>(env: Env<'a>, bytes: &[u8]) -> Binary<'a> {
    let mut b = NewBinary::new(env, bytes.len());
    b.as_mut_slice().copy_from_slice(bytes);
    b.into()
}

#[rustler::nif(schedule = "DirtyCpu")]
fn open(lib_path: String, data_root: String, batch_size: u32) -> Result<ResourceArc<BatchResource>, String> {
    let lib = unsafe { Library::new(&lib_path).map_err(|e| format!("load {}: {}", lib_path, e))? };
    let root = CString::new(data_root).map_err(|e| e.to_string())?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let code = unsafe { sym::<CreateFn>(&lib, b"msl_batch_create\0")?(root.as_ptr(), batch_size, &mut ptr) };
    check(&lib, code, "msl_batch_create")?;
    Ok(ResourceArc::new(BatchResource(Mutex::new(Batch { lib, ptr, size: batch_size as usize }))))
}

#[rustler::nif]
fn batch_size(res: ResourceArc<BatchResource>) -> usize {
    res.0.lock().unwrap().size
}

#[rustler::nif]
fn match_config_default<'a>(env: Env<'a>, res: ResourceArc<BatchResource>) -> Result<Binary<'a>, String> {
    let b = res.0.lock().unwrap();
    let cfg = unsafe { sym::<ConfigDefaultFn>(&b.lib, b"msl_match_config_default\0")?() };
    let bytes = unsafe { std::slice::from_raw_parts(&cfg as *const MslMatchConfig as *const u8, CONFIG_SIZE) };
    Ok(to_binary(env, bytes))
}

/// configs: batch_size x 52 bytes; mask: batch_size bytes or empty (= all).
#[rustler::nif(schedule = "DirtyCpu")]
fn reset<'a>(env: Env<'a>, res: ResourceArc<BatchResource>, configs: Binary, mask: Binary) -> Result<Binary<'a>, String> {
    let b = res.0.lock().unwrap();
    if configs.len() != b.size * CONFIG_SIZE {
        return Err(format!("configs must be {} x {} bytes, got {}", b.size, CONFIG_SIZE, configs.len()));
    }
    let mask_ptr = if mask.len() == b.size { mask.as_ptr() } else { std::ptr::null() };
    let mut obs = vec![0u8; b.size * OBS_SIZE];
    let code = unsafe { sym::<ResetFn>(&b.lib, b"msl_batch_reset\0")?(b.ptr, configs.as_ptr(), mask_ptr, obs.as_mut_ptr()) };
    check(&b.lib, code, "msl_batch_reset")?;
    Ok(to_binary(env, &obs))
}

/// inputs: batch_size x 32 bytes (raw MslInput). Returns {observations, terminals}.
#[rustler::nif(schedule = "DirtyCpu")]
fn step<'a>(env: Env<'a>, res: ResourceArc<BatchResource>, inputs: Binary) -> Result<(Binary<'a>, Binary<'a>), String> {
    let b = res.0.lock().unwrap();
    if inputs.len() != b.size * INPUT_SIZE {
        return Err(format!("inputs must be {} x {} bytes, got {}", b.size, INPUT_SIZE, inputs.len()));
    }
    let mut obs = vec![0u8; b.size * OBS_SIZE];
    let mut term = vec![0u8; b.size * TERM_SIZE];
    let code = unsafe { sym::<StepFn>(&b.lib, b"msl_batch_step\0")?(b.ptr, inputs.as_ptr(), obs.as_mut_ptr(), term.as_mut_ptr()) };
    check(&b.lib, code, "msl_batch_step")?;
    Ok((to_binary(env, &obs), to_binary(env, &term)))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn observe<'a>(env: Env<'a>, res: ResourceArc<BatchResource>) -> Result<(Binary<'a>, Binary<'a>), String> {
    let b = res.0.lock().unwrap();
    let mut obs = vec![0u8; b.size * OBS_SIZE];
    let mut term = vec![0u8; b.size * TERM_SIZE];
    let code = unsafe { sym::<ObserveFn>(&b.lib, b"msl_batch_observe\0")?(b.ptr, obs.as_mut_ptr(), term.as_mut_ptr()) };
    check(&b.lib, code, "msl_batch_observe")?;
    Ok((to_binary(env, &obs), to_binary(env, &term)))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn save<'a>(env: Env<'a>, res: ResourceArc<BatchResource>, index: u32) -> Result<Binary<'a>, String> {
    let b = res.0.lock().unwrap();
    let mut need: usize = 0;
    let code = unsafe { sym::<SaveSizeFn>(&b.lib, b"msl_batch_save_size\0")?(b.ptr, index, &mut need) };
    check(&b.lib, code, "msl_batch_save_size")?;
    let mut buf = vec![0u8; need];
    let mut written: usize = 0;
    let code = unsafe { sym::<SaveFn>(&b.lib, b"msl_batch_save\0")?(b.ptr, index, buf.as_mut_ptr(), need, &mut written) };
    check(&b.lib, code, "msl_batch_save")?;
    buf.truncate(written);
    Ok(to_binary(env, &buf))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn restore(res: ResourceArc<BatchResource>, index: u32, state: Binary) -> Result<rustler::Atom, String> {
    let b = res.0.lock().unwrap();
    let code = unsafe { sym::<RestoreFn>(&b.lib, b"msl_batch_restore\0")?(b.ptr, index, state.as_ptr(), state.len()) };
    check(&b.lib, code, "msl_batch_restore")?;
    Ok(rustler::types::atom::ok())
}

fn load(env: Env, _: rustler::Term) -> bool {
    rustler::resource!(BatchResource, env);
    true
}

rustler::init!("Elixir.ExPhil.Bridge.SimNif", load = load);
