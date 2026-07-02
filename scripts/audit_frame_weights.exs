# Audit: inspect frame_weights distribution under current vs. proposed criteria.
#
# Loads 5 replay files, applies both:
#   - current criterion (buttons + main stick)
#   - proposed criterion (buttons + main stick + c-stick + shoulder)
# and prints the action/neutral fraction under each.
#
# Self-contained: does not import Data (avoids EXLA at startup).
# Run with: mix run --no-start scripts/audit_frame_weights.exs

alias ExPhil.Data.Peppi

replay_dir = "./replays/huggingface"

paths =
  case File.ls(replay_dir) do
    {:ok, files} ->
      files
      |> Enum.filter(&String.ends_with?(&1, ".slp"))
      |> Enum.sort()
      |> Enum.take(5)
      |> Enum.map(&Path.join(replay_dir, &1))

    {:error, reason} ->
      IO.puts(:stderr, "Failed to list #{replay_dir}: #{inspect(reason)}")
      System.halt(1)
  end

IO.puts("Parsing #{length(paths)} replay files...")

# Discretization matches Data.discretize_axis (axis_buckets=17, center=8 at value=0.5).
# For 17 buckets: floor(0.5 * 17) = floor(8.5) = 8.
discretize_axis = fn value, buckets -> min(Kernel.trunc(:math.floor(value * buckets)), buckets - 1) end
discretize_shoulder = fn value, buckets -> min(Kernel.trunc(:math.floor(value * buckets)), buckets - 1) end

# Build action map matching the same shape compute_frame_weights expects.
controller_to_action = fn controller ->
  %{
    buttons: %{
      a: controller.button_a,
      b: controller.button_b,
      x: controller.button_x,
      y: controller.button_y,
      z: controller.button_z,
      l: controller.button_l,
      r: controller.button_r,
      d_up: controller.button_d_up
    },
    main_x: discretize_axis.(controller.main_stick_x, 17),
    main_y: discretize_axis.(controller.main_stick_y, 17),
    c_x: discretize_axis.(controller.c_stick_x, 17),
    c_y: discretize_axis.(controller.c_stick_y, 17),
    shoulder: discretize_shoulder.(Kernel.max(controller.l_trigger, controller.r_trigger), 4)
  }
end

# Inspect the first parsed replay's structure to find the right port.
{first_path, _rest} = List.pop_at(paths, 0)
{:ok, first_replay} = Peppi.parse(first_path, player_port: 1)
first_frame = hd(first_replay.frames)
ports = first_frame.players |> Map.keys() |> Enum.sort()
IO.puts("Ports present: #{inspect(ports)}; using first port: #{hd(ports)}")
target_port = hd(ports)

actions =
  paths
  |> Enum.flat_map(fn path ->
    case Peppi.parse(path, player_port: 1) do
      {:ok, replay} ->
        Enum.map(replay.frames, fn frame ->
          player = Map.get(frame.players, target_port)
          controller_to_action.(player.controller)
        end)

      {:error, reason} ->
        IO.puts(:stderr, "Skip #{Path.basename(path)}: #{inspect(reason)}")
        []
    end
  end)

n = length(actions)
IO.puts("Extracted #{n} action frames")

current = fn action ->
  buttons = action[:buttons]
  any_button = Enum.any?(buttons, fn {_k, v} -> v == true end)
  stick_moved = action[:main_x] != 8 or action[:main_y] != 8
  any_button or stick_moved
end

proposed = fn action ->
  buttons = action[:buttons]
  any_button = Enum.any?(buttons, fn {_k, v} -> v == true end)
  stick_moved = action[:main_x] != 8 or action[:main_y] != 8
  cstick_moved = action[:c_x] != 8 or action[:c_y] != 8
  shoulder_pressed = action[:shoulder] != 0
  any_button or stick_moved or cstick_moved or shoulder_pressed
end

{cur_action, cur_neutral} = Enum.split_with(actions, current)
{prop_action, prop_neutral} = Enum.split_with(actions, proposed)

pct = fn count, total -> Float.round(count / total * 100, 2) end

IO.puts("\n=== Current criterion (buttons + main stick) ===")
IO.puts("  action  frames: #{length(cur_action)} (#{pct.(length(cur_action), n)}%)")
IO.puts("  neutral frames: #{length(cur_neutral)} (#{pct.(length(cur_neutral), n)}%)")

IO.puts("\n=== Proposed criterion (+ c-stick + shoulder) ===")
IO.puts("  action  frames: #{length(prop_action)} (#{pct.(length(prop_action), n)}%)")
IO.puts("  neutral frames: #{length(prop_neutral)} (#{pct.(length(prop_neutral), n)}%)")

flipped = Enum.filter(actions, fn a -> not current.(a) and proposed.(a) end)

IO.puts("\n=== Delta: frames newly flagged as action ===")
IO.puts("  flipped: #{length(flipped)} (#{pct.(length(flipped), n)}% of all frames)")

cstick_count = Enum.count(flipped, fn a -> a[:c_x] != 8 or a[:c_y] != 8 end)
shoulder_count = Enum.count(flipped, fn a -> a[:shoulder] != 0 end)

IO.puts("    of which c-stick moved: #{cstick_count}")
IO.puts("    of which shoulder pressed: #{shoulder_count}")
