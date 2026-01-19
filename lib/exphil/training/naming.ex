defmodule ExPhil.Training.Naming do
  @moduledoc """
  Generates memorable names for model checkpoints.

  Creates Docker-style random names like "wavedashing_falcon" or "phob_sweetspot"
  by combining adjectives with nouns. Includes three categories of terms:

  ## Categories

  ### General (50% chance)
  - Adjectives: brave, cosmic, swift, legendary, quantum, etc.
  - Nouns: falcon, phoenix, dragon, wolf, etc. (animals)

  ### Melee Tech (30% chance)
  - Adjectives: wavedashing, multishining, techchasing, lcanceling, etc.
  - Nouns: tipper, sweetspot, wavedash, shine, meteor, etc.

  ### Hardware/Mods (20% chance)
  - Adjectives: notched, modded, phob, rollbacked, homebrewed, etc.
  - Nouns: phob, goomwave, rectangle, boxx, slippi, ucf, etc.

  ## Examples

      iex> Naming.generate()
      "wavedashing_falcon"

      iex> Naming.generate()
      "notched_phob"

  """

  # General adjectives (positive/neutral vibes)
  @adjectives ~w(
    ancient bold brave bright calm clever confident cosmic crispy crystal
    dapper daring dashing determined elegant epic fancy fast fearless fierce
    flaming fluffy focused friendly funky gentle glowing golden graceful grand
    happy hasty hidden humble hungry hyper icy jolly keen lazy legendary
    lively lucky majestic mellow mighty misty mysterious nifty noble optimal
    peaceful patient phantom pixelated polished powerful precise proud quantum
    quick quirky radical rapid ready rising robust royal rusty sacred savage
    serene sharp shiny silent silly sleek sleepy slick smooth sneaky snowy
    solid speedy spicy spiffy stellar stoic sturdy subtle super swift tactical
    tender thirsty thunder tough tranquil turbo ultra upbeat valiant vibrant
    vivid wandering warm wicked wild wise witty zany zen zippy
  )

  # Melee-specific adjectives (tech skill / gameplay terms)
  # Sources: SmashWiki, Melee Library, Smashboards
  @melee_adjectives ~w(
    wavedashing dashdancing ledgedashing moonwalking techchasing
    chaingrabbing edgeguarding shielddropping lcanceling multishining
    pillaring wobbling crouch_canceling powershielding wavelanding
    shffling meteor_canceling pivot_grabbing ledgehopping platform_dropping
    shield_stopping dashjumping shield_grabbing ledgestalling
    needle_turnaround haxdashing sdi_ing asdi_ing double_jumping
    fastfalling shorthopping perfect_wavelanding amsah_teching
    shine_spiking gentlemaning uptilting downsmashing
  )

  # Hardware/mod scene adjectives (controller mods, custom hardware, scene tech)
  # Sources: Rectangle controllers, PhobGCC, Slippi, texture packs
  @hardware_adjectives ~w(
    notched modded flashed homebrewed rectangular digital
    rollbacked slippi phob goomwaved cardboarded snapbacked
    pode triggered latched calibrated gated octagated textured
    reskinned softmodded hardmodded overclocked undervolted
  )

  # Melee concept nouns (non-character terms to avoid confusion)
  @melee_nouns ~w(
    combo edgeguard gimpal punish neutral spacing tipper sweetspot
    sourspot hitbox hurtbox knockback hitstun shieldstun grabrange
    dashback pivot wavedash waveland ledgedash moonwalk shine
    reflector rest meteor spike dair bair fair uair nair zair
    jab tilt smash grab throw pummel shield spotdodge airdodge
    recovery gimp edgehog ledgehop platform teeter crouch
    ftilt utilt dtilt fsmash usmash dsmash fthrow bthrow uthrow dthrow
  )

  # Hardware/mod scene nouns (controllers, mods, tools)
  # Sources: Rectangle controllers, PhobGCC, Wii hacking, texture packs
  @hardware_nouns ~w(
    phob goomwave rectangle boxx frame1 smashbox lbx
    notch gate octagon stickbox potentiometer trigger snapback
    rollback netcode buffer nintendont portalpod wii gamecube
    adapter mayflash raphnet iso nand priiloader homebrew
    geckocode gecko bootmii usbloader slippi replay savestate
    ucf dashback texturepack reskin twentyxx unclepunch
  )

  # General nouns (animals, objects)
  @nouns ~w(
    badger bat bear beaver bird bison buffalo cat cheetah cobra coyote
    crane crow deer dolphin dragon eagle elephant falcon ferret finch
    flamingo frog gazelle gecko giraffe goat goose gorilla grasshopper
    hamster hawk hedgehog heron hippo horse hound hyena iguana impala
    jackal jaguar jellyfish kangaroo koala lemur leopard lion lizard llama
    lobster lynx mammoth mantis meerkat mongoose monkey moose moth mouse
    newt octopus orca osprey ostrich otter owl panda panther parrot peacock
    pelican penguin phoenix pigeon piranha pony porcupine puma rabbit
    raccoon raven rhino robin salmon scorpion seahorse seal shark sheep
    sloth snail snake sparrow sphinx spider squid squirrel stallion stingray
    stork swan tiger toad toucan trout turkey turtle viper vulture walrus
    weasel whale wolf wolverine wombat woodpecker yak zebra
  )

  @doc """
  Generate a random memorable name.

  Returns a string like "wavedashing_falcon" or "modded_phob".
  Uses weighted random selection across general, Melee, and hardware categories.

  ## Examples

      iex> Naming.generate()
      "brave_phoenix"

  """
  @spec generate() :: String.t()
  def generate do
    adj = pick_adjective()
    noun = pick_noun()
    "#{adj}_#{noun}"
  end

  @doc """
  Generate a name with a specific seed for reproducibility.

  ## Examples

      iex> Naming.generate(12345)
      "cosmic_falcon"

      iex> Naming.generate(12345)
      "cosmic_falcon"

  """
  @spec generate(integer()) :: String.t()
  def generate(seed) when is_integer(seed) do
    :rand.seed(:exsss, {seed, seed, seed})
    generate()
  end

  # Pick adjective: 50% general, 30% Melee tech, 20% hardware/mod
  defp pick_adjective do
    roll = :rand.uniform(100)

    cond do
      roll <= 50 -> Enum.random(@adjectives)
      roll <= 80 -> Enum.random(@melee_adjectives)
      true -> Enum.random(@hardware_adjectives)
    end
  end

  # Pick noun: 50% general, 30% Melee concept, 20% hardware/mod
  defp pick_noun do
    roll = :rand.uniform(100)

    cond do
      roll <= 50 -> Enum.random(@nouns)
      roll <= 80 -> Enum.random(@melee_nouns)
      true -> Enum.random(@hardware_nouns)
    end
  end

  @doc """
  Get all available adjectives (for testing/inspection).
  """
  def adjectives, do: @adjectives ++ @melee_adjectives ++ @hardware_adjectives

  @doc """
  Get all available nouns (for testing/inspection).
  """
  def nouns, do: @nouns ++ @melee_nouns ++ @hardware_nouns
end
