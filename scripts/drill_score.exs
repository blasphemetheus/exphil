# Hit-confirm drill scorer (DRILL_HITCONFIRM.md — the authoritative pass).
#
# Rescores a drill bank (scripts/drill_episode.exs output) from the .slp
# replays using the EXACT drill_table_mine detector (hitstun OR thrown
# 239..243 OR captured 223..232 rising edges), anchored at the recorded
# handoff frames in episodes.jsonl. The live counter in the driver is a
# convenience; this is the number that goes in a RESULTS file.
#
#   mix run scripts/drill_score.exs --bank eval_runs/0902_drill_smoke \
#     --out eval_runs/0902_drill_smoke/RESULTS.md
#
# Options: --window (default: each episode's recorded window via end_frame,
# else 240) · --out PATH

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [bank: :string, window: :integer, out: :string])

bank = opts[:bank] || raise "--bank required"

# The bank's manifest snapshot (drill.json) pins bands and references;
# older banks fall back to the Drill 1 manifest in drills/.
drill =
  [Path.join(bank, "drill.json"), "drills/uthrow_low_mid.json"]
  |> Enum.find(&File.exists?/1)
  |> case do
    nil -> %{}
    p -> p |> File.read!() |> Jason.decode!()
  end

window = opts[:window] || drill["window"] || 240
bands = for [lo, hi] <- (drill["cell"] || %{})["bands"] || [[0, 19]], do: {lo, hi}
references = drill["references"] || %{}

band_of = fn pct ->
  case Enum.find(bands, fn {lo, hi} -> pct >= lo and pct < hi + 1 end) do
    nil -> "out-of-band"
    {lo, hi} -> "#{lo}-#{hi}"
  end
end

jsonl = Path.join(bank, "episodes.jsonl")

rows =
  jsonl
  |> File.read!()
  |> String.split("\n", trim: true)
  |> Enum.map(&Jason.decode!/1)
  |> Enum.filter(& &1["slp"])

Output.banner("Drill bank scorer")
Output.puts("  #{length(rows)} episodes across #{rows |> Enum.map(& &1["slp"]) |> Enum.uniq() |> length()} replays")

hitstun? = fn pl ->
  pl != nil and
    ((pl.hitstun_frames_left || 0) > 0 or pl.action in 239..243 or pl.action in 223..232)
end

# The driver records the live bot on port 1 and the victim on port 2, so the
# remap is an identity — passed anyway to satisfy the port-convention stamp.
states_by_slp =
  rows
  |> Enum.map(& &1["slp"])
  |> Enum.uniq()
  |> Map.new(fn slp ->
    case Peppi.parse(Path.join(bank, slp)) do
      {:ok, replay} ->
        states =
          replay
          |> Peppi.to_training_frames(player_port: 1, opponent_port: 2, remap_ports: true)
          |> Enum.map(& &1.game_state)

        {slp, {List.to_tuple(states), Map.new(Enum.with_index(states), fn {s, i} -> {s.frame, i} end)}}

      err ->
        Output.warning("  parse failed for #{slp}: #{inspect(err)}")
        {slp, nil}
    end
  end)

{scored, anchor_mismatches} =
  Enum.reduce(rows, {[], 0}, fn row, {acc, mism} ->
    case states_by_slp[row["slp"]] do
      nil ->
        {acc, mism}

      {states_t, index} ->
        n = tuple_size(states_t)

        case index[row["handoff"]] do
          nil ->
            Output.warning("  handoff frame #{row["handoff"]} not found in #{row["slp"]}")
            {acc, mism}

          t0 ->
            own0 = elem(states_t, t0).players[1]
            vic0 = elem(states_t, t0).players[2]
            # Calibration guard: the anchor should be the throw itself.
            mism = if own0 && own0.action in 219..222, do: mism, else: mism + 1

            {hits, _} =
              Enum.reduce(t0..min(t0 + window, n - 1), {0, true}, fn t, {h, prev_hs} ->
                hs = hitstun?.(elem(states_t, t).players[2])
                {(if hs and not prev_hs, do: h + 1, else: h), hs}
              end)

            vic_end = elem(states_t, min(t0 + window, n - 1)).players[2]

            dmg =
              if vic_end && vic0,
                do: max((vic_end.percent || 0.0) - (vic0.percent || 0.0), 0.0),
                else: 0.0

            stock =
              if vic_end && vic0 && (vic_end.stock || 0) < (vic0.stock || 0), do: 1, else: 0

            band = row["band"] || band_of.(row["vic_pct0"] || 0.0)

            {[%{band: band, hits: hits + 1, dmg: dmg, stock: stock,
                live_hits: row["hits"], live_dmg: row["dmg"]} | acc],
             mism}
        end
    end
  end)

if scored == [] do
  Output.error("nothing scored")
  System.halt(1)
end

n = length(scored)

live_drift =
  scored
  |> Enum.count(fn s -> s.live_hits != nil and s.live_hits != s.hits end)

f1 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 1) end

band_rows =
  scored
  |> Enum.group_by(& &1.band)
  |> Enum.sort()
  |> Enum.flat_map(fn {band, eps} ->
    bn = length(eps)
    mh = Enum.sum(Enum.map(eps, & &1.hits)) / bn
    md = Enum.sum(Enum.map(eps, & &1.dmg)) / bn
    deep = Enum.count(eps, &(&1.hits >= 3)) / bn
    stocks = Enum.sum(Enum.map(eps, & &1.stock))

    hist =
      eps
      |> Enum.frequencies_by(& &1.hits)
      |> Enum.sort()
      |> Enum.map_join("  ", fn {h, c} -> "#{h}:#{c}" end)

    bot =
      "| bot #{band}% | #{bn} | #{f1.(mh)} | #{round(deep * 100)} | #{f1.(md)} | #{stocks} |"

    ref =
      case references[band] do
        %{"hits" => h, "deep3" => d3, "dmg" => d, "n" => rn} ->
          ["| expert #{band}% | #{rn} | #{h} | #{d3} | #{d} | 0 |"]

        _ ->
          ["| expert #{band}% | - | (no reference mined) | | | |"]
      end

    ["#{bot}"] ++ ref ++ ["", "#{band}% hits histogram: #{hist}", ""]
  end)

report = """
# Drill bank score — #{bank} (#{drill["name"] || "unknown drill"})

#{n} episodes scored (window #{window} f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): #{anchor_mismatches}. Live-counter
disagreements: #{live_drift}/#{n}.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
#{band_rows |> Enum.filter(&String.starts_with?(&1, "|")) |> Enum.join("\n")}

#{band_rows |> Enum.reject(&String.starts_with?(&1, "|")) |> Enum.join("\n")}
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
