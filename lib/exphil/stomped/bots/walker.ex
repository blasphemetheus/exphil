defmodule ExPhil.Stomped.Bots.Walker do
  @moduledoc """
  The arena's quickstart bot: walk at the opponent and tap A in reach.

  It reads two fields of the stream (each player's x position) and exists to
  prove the server end to end — the deploy check, the steps, pad timing —
  before a policy is wired in. It is not meant to win.

  Options: `:character` (slp external id, default 2 = Fox), `:costume`.
  """

  @behaviour ExPhil.Stomped.Bot

  alias ExPhil.Stomped.Events

  # Close enough to hit, in game units.
  @reach 10
  # A walking pair closes about this many units a frame.
  @closing_speed 3

  @impl true
  def init(opts) do
    %{
      character: Keyword.get(opts, :character, 2),
      costume: Keyword.get(opts, :costume, 0),
      x: %{}
    }
  end

  @impl true
  def pick(_step, state) do
    {%{character: state.character, costume: state.costume}, state}
  end

  @impl true
  def handle_frame(_frame, events, ctx, state) do
    post_frame = Events.post_frame()

    x =
      Enum.reduce(events, state.x, fn
        # Post-Frame Update: i32 frame · u8 port · u8 follower · u8 char · u16 action · f32 x
        {^post_frame, <<_frame::32, port, 0, _char, _action::16, x::big-float-32, _::binary>>},
        acc ->
          Map.put(acc, port, x)

        _event, acc ->
          acc
      end)

    state = %{state | x: x}

    pad =
      case {x[ctx.port], x[1 - ctx.port]} do
        {mine, theirs} when is_number(mine) and is_number(theirs) ->
          dx = theirs - mine

          # The pad lands `lead` frames from now; the gap will have closed by then.
          if abs(dx) < @reach + @closing_speed * ctx.lead do
            if rem(ctx.target, 2) != 0, do: %{buttons: [:a]}, else: %{}
          else
            %{stick_x: if(dx > 0, do: 188, else: 68)}
          end

        _ ->
          %{}
      end

    {pad, state}
  end
end
