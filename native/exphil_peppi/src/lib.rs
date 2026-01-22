//! ExPhil Peppi NIF
//!
//! Rust NIF for parsing Slippi replay files using the Peppi library.
//! Provides fast, native parsing of .slp files for training data extraction.

use peppi::game::Game;
use peppi::frame::transpose::{Frame, Data, Pre};
use peppi::io::slippi;
use rustler::{Atom, NifResult, NifStruct};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

mod atoms {
    rustler::atoms! {
        ok,
        error,
        nil,
    }
}

// ============================================================================
// NIF Structs - These map directly to Elixir structs
// ============================================================================

/// Parsed controller state
#[derive(Debug, NifStruct)]
#[module = "ExPhil.Data.Peppi.Controller"]
pub struct Controller {
    pub main_stick_x: f64,
    pub main_stick_y: f64,
    pub c_stick_x: f64,
    pub c_stick_y: f64,
    pub l_trigger: f64,
    pub r_trigger: f64,
    pub button_a: bool,
    pub button_b: bool,
    pub button_x: bool,
    pub button_y: bool,
    pub button_z: bool,
    pub button_l: bool,
    pub button_r: bool,
    pub button_start: bool,
    pub button_d_up: bool,
    pub button_d_down: bool,
    pub button_d_left: bool,
    pub button_d_right: bool,
}

/// Parsed player state for a single frame
#[derive(Debug, NifStruct)]
#[module = "ExPhil.Data.Peppi.PlayerFrame"]
pub struct PlayerFrame {
    pub character: i32,
    pub x: f64,
    pub y: f64,
    pub percent: f64,
    pub stock: i32,
    pub facing: i32,
    pub action: u16,
    pub action_frame: f64,
    pub invulnerable: bool,
    pub jumps_left: i32,
    pub on_ground: bool,
    pub shield_strength: f64,
    pub hitstun_frames_left: f64,
    pub speed_air_x_self: f64,
    pub speed_ground_x_self: f64,
    pub speed_y_self: f64,
    pub speed_x_attack: f64,
    pub speed_y_attack: f64,
    pub controller: Controller,
}

/// A single parsed game frame
#[derive(Debug, NifStruct)]
#[module = "ExPhil.Data.Peppi.GameFrame"]
pub struct GameFrame {
    pub frame_number: i32,
    pub players: HashMap<i32, PlayerFrame>,
}

/// Player metadata from game start
#[derive(Debug, NifStruct)]
#[module = "ExPhil.Data.Peppi.PlayerMeta"]
pub struct PlayerMeta {
    pub port: i32,
    pub character: i32,
    pub character_name: String,
    pub tag: Option<String>,
}

/// Replay metadata
#[derive(Debug, NifStruct)]
#[module = "ExPhil.Data.Peppi.ReplayMeta"]
pub struct ReplayMeta {
    pub path: String,
    pub stage: i32,
    pub duration_frames: i32,
    pub players: Vec<PlayerMeta>,
}

/// Complete parsed replay
#[derive(Debug, NifStruct)]
#[module = "ExPhil.Data.Peppi.ParsedReplay"]
pub struct ParsedReplay {
    pub frames: Vec<GameFrame>,
    pub metadata: ReplayMeta,
}

// ============================================================================
// Character & Stage Mappings
// ============================================================================

/// Convert peppi's external character ID to our standard ID
fn character_id(char_id: u8) -> i32 {
    // Peppi uses "external" character IDs (CSS order)
    // Map to our consistent ordering for embeddings
    match char_id {
        0x00 => 0,   // Captain Falcon
        0x01 => 1,   // Donkey Kong
        0x02 => 2,   // Fox
        0x03 => 3,   // Game & Watch
        0x04 => 4,   // Kirby
        0x05 => 5,   // Bowser
        0x06 => 6,   // Link
        0x07 => 7,   // Luigi
        0x08 => 8,   // Mario
        0x09 => 9,   // Marth
        0x0A => 10,  // Mewtwo
        0x0B => 11,  // Ness
        0x0C => 12,  // Peach
        0x0D => 13,  // Pikachu
        0x0E => 14,  // Ice Climbers (Popo)
        0x0F => 15,  // Jigglypuff
        0x10 => 16,  // Samus
        0x11 => 17,  // Yoshi
        0x12 => 18,  // Zelda
        0x13 => 19,  // Sheik
        0x14 => 20,  // Falco
        0x15 => 21,  // Young Link
        0x16 => 22,  // Dr. Mario
        0x17 => 23,  // Roy
        0x18 => 24,  // Pichu
        0x19 => 25,  // Ganondorf
        0x20 => 14,  // Ice Climbers (Nana) - same as Popo
        _ => -1,
    }
}

fn character_name(char_id: u8) -> String {
    // Peppi uses "external" character IDs (CSS order), not internal game IDs
    // See: https://github.com/hohav/peppi/blob/main/src/ssbm.rs
    match char_id {
        0x00 => "Captain Falcon",
        0x01 => "Donkey Kong",
        0x02 => "Fox",
        0x03 => "Game & Watch",
        0x04 => "Kirby",
        0x05 => "Bowser",
        0x06 => "Link",
        0x07 => "Luigi",
        0x08 => "Mario",
        0x09 => "Marth",
        0x0A => "Mewtwo",
        0x0B => "Ness",
        0x0C => "Peach",
        0x0D => "Pikachu",
        0x0E => "Ice Climbers",  // Popo
        0x0F => "Jigglypuff",
        0x10 => "Samus",
        0x11 => "Yoshi",
        0x12 => "Zelda",
        0x13 => "Sheik",
        0x14 => "Falco",
        0x15 => "Young Link",
        0x16 => "Dr. Mario",
        0x17 => "Roy",
        0x18 => "Pichu",
        0x19 => "Ganondorf",
        0x1A => "Master Hand",
        0x1B => "Crazy Hand",
        0x1C => "Wire Frame Male",
        0x1D => "Wire Frame Female",
        0x1E => "Giga Bowser",
        0x1F => "Sandbag",
        0x20 => "Ice Climbers",  // Nana (solo)
        _ => "Unknown",
    }.to_string()
}

/// Convert stage ID to our standard ID
fn stage_id(stage: u16) -> i32 {
    // Standard Melee stage IDs
    match stage {
        2 => 2,   // Fountain of Dreams
        3 => 3,   // Pokemon Stadium
        8 => 8,   // Yoshi's Story
        28 => 28, // Dream Land N64
        31 => 31, // Battlefield
        32 => 32, // Final Destination
        _ => stage as i32,
    }
}

// ============================================================================
// Parsing Functions
// ============================================================================

fn parse_controller(pre: &Pre) -> Controller {
    let joystick = &pre.joystick;
    let cstick = &pre.cstick;
    let triggers = &pre.triggers_physical;
    let buttons = pre.buttons_physical;

    // Normalize from [-1, 1] to [0, 1]
    let main_x = (joystick.x as f64 + 1.0) / 2.0;
    let main_y = (joystick.y as f64 + 1.0) / 2.0;
    let c_x = (cstick.x as f64 + 1.0) / 2.0;
    let c_y = (cstick.y as f64 + 1.0) / 2.0;

    // Button masks for physical buttons (16-bit)
    const A: u16 = 0x0100;
    const B: u16 = 0x0200;
    const X: u16 = 0x0400;
    const Y: u16 = 0x0800;
    const Z: u16 = 0x0010;
    const L: u16 = 0x0040;
    const R: u16 = 0x0020;
    const START: u16 = 0x1000;
    const D_UP: u16 = 0x0008;
    const D_DOWN: u16 = 0x0004;
    const D_LEFT: u16 = 0x0001;
    const D_RIGHT: u16 = 0x0002;

    Controller {
        main_stick_x: main_x,
        main_stick_y: main_y,
        c_stick_x: c_x,
        c_stick_y: c_y,
        l_trigger: triggers.l as f64,
        r_trigger: triggers.r as f64,
        button_a: (buttons & A) != 0,
        button_b: (buttons & B) != 0,
        button_x: (buttons & X) != 0,
        button_y: (buttons & Y) != 0,
        button_z: (buttons & Z) != 0,
        button_l: (buttons & L) != 0,
        button_r: (buttons & R) != 0,
        button_start: (buttons & START) != 0,
        button_d_up: (buttons & D_UP) != 0,
        button_d_down: (buttons & D_DOWN) != 0,
        button_d_left: (buttons & D_LEFT) != 0,
        button_d_right: (buttons & D_RIGHT) != 0,
    }
}

fn parse_player_frame(data: &Data) -> PlayerFrame {
    let pre = &data.pre;
    let post = &data.post;

    // Direction is stored as f32: negative = left, positive = right
    let facing = if post.direction < 0.0 { -1 } else { 1 };

    // Extract velocities if available
    let (speed_air_x, speed_ground_x, speed_y, speed_x_attack, speed_y_attack) =
        if let Some(ref velocities) = post.velocities {
            (
                velocities.self_x_air as f64,
                velocities.self_x_ground as f64,
                velocities.self_y as f64,
                velocities.knockback_x as f64,
                velocities.knockback_y as f64,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

    // Check invulnerability from state flags
    let invulnerable = post.state_flags
        .map(|flags| {
            // StateFlags is a tuple struct with 5 u8 values
            // Check specific flag bits for invulnerability
            (flags.0 & 0x04) != 0 || (flags.0 & 0x10) != 0
        })
        .unwrap_or(false);

    // Check if grounded
    let on_ground = post.airborne
        .map(|a| a == 0)
        .unwrap_or(true);

    PlayerFrame {
        character: character_id(post.character),
        x: post.position.x as f64,
        y: post.position.y as f64,
        percent: post.percent as f64,
        stock: post.stocks as i32,
        facing,
        action: post.state,
        action_frame: post.state_age.unwrap_or(0.0) as f64,
        invulnerable,
        jumps_left: post.jumps.unwrap_or(2) as i32,
        on_ground,
        shield_strength: post.shield as f64,
        hitstun_frames_left: post.hitlag.unwrap_or(0.0) as f64,
        speed_air_x_self: speed_air_x,
        speed_ground_x_self: speed_ground_x,
        speed_y_self: speed_y,
        speed_x_attack: speed_x_attack,
        speed_y_attack: speed_y_attack,
        controller: parse_controller(pre),
    }
}

fn parse_frame(frame: &Frame) -> GameFrame {
    let mut players = HashMap::new();

    for port_data in &frame.ports {
        let port_num = port_data.port as i32 + 1;  // Convert 0-indexed to 1-indexed
        let player_frame = parse_player_frame(&port_data.leader);
        players.insert(port_num, player_frame);
    }

    GameFrame {
        frame_number: frame.id,
        players,
    }
}

fn parse_game<G: Game>(game: &G, path: &str) -> ParsedReplay {
    // Extract metadata from game start
    let start = game.start();
    let stage = stage_id(start.stage);

    let mut player_metas = Vec::new();
    for player in &start.players {
        player_metas.push(PlayerMeta {
            port: player.port as i32 + 1,
            character: character_id(player.character),
            character_name: character_name(player.character),
            tag: player.name_tag.as_ref().map(|s| s.0.clone()),
        });
    }

    // Parse all frames
    let num_frames = game.len();
    let mut frames = Vec::with_capacity(num_frames);

    for idx in 0..num_frames {
        let frame = game.frame(idx);
        frames.push(parse_frame(&frame));
    }

    let metadata = ReplayMeta {
        path: path.to_string(),
        stage,
        duration_frames: frames.len() as i32,
        players: player_metas,
    };

    ParsedReplay { frames, metadata }
}

// ============================================================================
// NIF Functions
// ============================================================================

/// Parse a single .slp replay file
#[rustler::nif]
fn parse_replay(path: String) -> NifResult<(Atom, ParsedReplay)> {
    let file = File::open(&path)
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to open file: {}", e))))?;

    let mut reader = BufReader::new(file);
    let game = slippi::read(&mut reader, None)
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to parse replay: {}", e))))?;

    let parsed = parse_game(&game, &path);
    Ok((atoms::ok(), parsed))
}

/// Parse a replay and return only frames for a specific player port
#[rustler::nif]
fn parse_replay_for_port(path: String, player_port: i32) -> NifResult<(Atom, ParsedReplay)> {
    let file = File::open(&path)
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to open file: {}", e))))?;

    let mut reader = BufReader::new(file);
    let game = slippi::read(&mut reader, None)
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to parse replay: {}", e))))?;

    let mut parsed = parse_game(&game, &path);

    // Filter to only include frames where the specified port has data
    parsed.frames.retain(|f| f.players.contains_key(&player_port));

    Ok((atoms::ok(), parsed))
}

/// Get replay metadata without parsing all frames (faster for filtering)
#[rustler::nif]
fn get_replay_metadata(path: String) -> NifResult<(Atom, ReplayMeta)> {
    let file = File::open(&path)
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to open file: {}", e))))?;

    let mut reader = BufReader::new(file);
    let game = slippi::read(&mut reader, None)
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to parse replay: {}", e))))?;

    let start = game.start();
    let stage = stage_id(start.stage);

    let mut player_metas = Vec::new();
    for player in &start.players {
        player_metas.push(PlayerMeta {
            port: player.port as i32 + 1,
            character: character_id(player.character),
            character_name: character_name(player.character),
            tag: player.name_tag.as_ref().map(|s| s.0.clone()),
        });
    }

    let metadata = ReplayMeta {
        path: path.to_string(),
        stage,
        duration_frames: game.len() as i32,
        players: player_metas,
    };

    Ok((atoms::ok(), metadata))
}

// Register NIFs
rustler::init!("Elixir.ExPhil.Data.Peppi");
