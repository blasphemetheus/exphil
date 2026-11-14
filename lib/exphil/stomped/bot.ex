defmodule ExPhil.Stomped.Bot do
  @moduledoc """
  What a bot has to answer to play a match on the stomped.dev arena.

  `ExPhil.Stomped.Match` owns the protocol (steps, pad framing, timing); a
  bot only decides. One bot state lives for one match socket.

  The step maps are the arena's JSON, decoded with string keys (`"port"`,
  `"stage"`, `"opponent"`, `"players"`, `"inputDelayFrames"`, ...).
  """

  alias ExPhil.Stomped.{Events, Pad}

  @typedoc """
  Per-frame context.

    * `:port` — our Slippi player index (0 or 1)
    * `:lead` — frames between the frame just seen and the pad being asked for
    * `:target` — the frame the returned pad will be polled at (`frame + lead`)
    * `:live?` — `false` while catching up through a `replay` answer; the pad
      is discarded, but stateful bots still need to see every frame
  """
  @type ctx :: %{port: 0 | 1, lead: pos_integer(), target: integer(), live?: boolean()}

  @doc "Start a bot for one match socket. `opts` come from the server config."
  @callback init(opts :: keyword()) :: state :: term()

  @doc "The `character` step: pick a character (slp external id) and costume. 5 s deadline."
  @callback pick(step :: map(), state :: term()) ::
              {%{character: 0..25, costume: 0..5}, state :: term()}

  @doc "The `ready` step: both picks are known. Prepare for the matchup. 10 s deadline."
  @callback ready(step :: map(), state :: term()) :: state :: term()

  @doc "Events outside any frame: the stream header (Game Start) and Game End."
  @callback handle_events(events :: [Events.event()], state :: term()) :: state :: term()

  @doc """
  One finished frame. Return the pad the game should poll at `ctx.target`.

  This runs inside the frame budget (11 ms from the arena's send to the
  pad's arrival, network included), so it must not block.
  """
  @callback handle_frame(
              frame :: integer(),
              events :: [Events.event()],
              ctx :: ctx(),
              state :: term()
            ) :: {Pad.inputs(), state :: term()}

  @doc "The `over` step: the result. The socket closes after it."
  @callback over(step :: map(), state :: term()) :: term()

  @optional_callbacks ready: 2, handle_events: 2, over: 2

  @characters %{
    "captain_falcon" => 0,
    "donkey_kong" => 1,
    "fox" => 2,
    "game_and_watch" => 3,
    "kirby" => 4,
    "bowser" => 5,
    "link" => 6,
    "luigi" => 7,
    "mario" => 8,
    "marth" => 9,
    "mewtwo" => 10,
    "ness" => 11,
    "peach" => 12,
    "pikachu" => 13,
    "ice_climbers" => 14,
    "jigglypuff" => 15,
    "samus" => 16,
    "yoshi" => 17,
    "zelda" => 18,
    "sheik" => 19,
    "falco" => 20,
    "young_link" => 21,
    "dr_mario" => 22,
    "roy" => 23,
    "pichu" => 24,
    "ganondorf" => 25
  }

  @doc "The arena's character ids (slp external ids) by name."
  @spec characters() :: %{String.t() => 0..25}
  def characters, do: @characters

  @doc "Character id for a name such as `\"fox\"`, or for an id given as a string."
  @spec character_id(String.t() | integer()) :: {:ok, 0..25} | :error
  def character_id(id) when is_integer(id) and id in 0..25, do: {:ok, id}

  def character_id(name) when is_binary(name) do
    key = name |> String.downcase() |> String.replace(~r/[\s-]+/, "_")

    case Map.fetch(@characters, key) do
      {:ok, id} ->
        {:ok, id}

      :error ->
        case Integer.parse(name) do
          {id, ""} when id in 0..25 -> {:ok, id}
          _ -> :error
        end
    end
  end

  def character_id(_), do: :error
end
