defmodule ExPhil.Training.MarginSampling do
  @moduledoc """
  Frame weights for the AERIAL-SHINE DECISION — the per-link Bernoulli the
  2026-08-05 delay-break study named as the residual human gap (chains die
  when one link's aerial B misses its window and the cycle balloons into a
  full hop; declared delay-id sets the decision margin: p10 −0.52 / +0.09
  / +0.25 at ids 2/3/4).

  Upweighting BCE on exactly these frames buys logit margin where the
  failure lives — the training-time analog of the delay-id effect, riding
  the same `sampling_weights` machinery as conversion/opener weighting
  (composed by elementwise max, no multiplicative blowup).

  Critical frame := the B PRESS EDGE (b true, previous frame b false)
  while in jumpsquat or aerial-jump family — the one decision frame per
  cycle whose margin the delay-id sweep measured. Measured on the
  canonical d1 fixture: the cycle has NO :aerial_jump dwell at all (the
  press happens DURING jumpsquat, shine materializes on the first
  airborne frame), and Peppi's action_frame convention breaks af-based
  cuts — hence family+edge, not family+af (caught by the smoke run,
  2026-08-05).
  """

  alias ExPhil.Eval.ShineChain

  @doc """
  Returns `{weights, stats}`; `stats = %{frames, upweighted}`.
  Frame lists are the SHIFTED lists (labels aligned to states), same
  contract as ConversionSampling/OpenerSampling.
  """
  def frame_weights(frame_lists, weight) when is_number(weight) do
    weights =
      Enum.flat_map(frame_lists, fn frames ->
        frames
        |> Enum.map_reduce(false, fn f, prev_b ->
          p = f.game_state.players[1]
          fam = ShineChain.family(trunc((p && p.action) || 0))
          b = f.controller.button_b

          # :air_reflect included: in delay-shifted lists the press edge
          # coincides with the reflector state it produces (measured on the
          # d1 fixture: all 787 aerial-press edges land on :air_reflect
          # frames; jumpsquat only ever carries the RELEASE edge).
          critical? = b and not prev_b and fam in [:jumpsquat, :aerial_jump, :air_reflect]
          {if(critical?, do: weight * 1.0, else: 1.0), b}
        end)
        |> elem(0)
      end)

    {weights, %{frames: length(weights), upweighted: Enum.count(weights, &(&1 != 1.0))}}
  end
end
