# AI-generated; attribution: docs/guides/MAMBA_BATCH_PARITY.md.
ExUnit.start(exclude: [:slow, :integration, :external], max_cases: 1)

if System.get_env("RELOAD_PLAYER") == "1",
  do: Code.compile_file("lib/exphil/embeddings/player.ex")

if System.get_env("RELOAD_PLAYER") == "1",
  do: Code.compile_file("lib/exphil/embeddings/game/spatial.ex")

Code.require_file("test/exphil/embeddings/player_frame_parity_test.exs")
Code.require_file("test/exphil/embeddings/player_test.exs")
Code.require_file("test/exphil/embeddings/game_test.exs")
