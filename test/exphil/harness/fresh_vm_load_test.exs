defmodule ExPhil.Harness.FreshVmLoadTest do
  @moduledoc """
  Loads a real checkpoint in a **separate VM** that never loaded the
  exporting module.

  GOTCHAS #68 cost ~3h and two full probe phases: every r11 probe crashed
  at agent load with "invalid or unsafe external representation of a term",
  because `binary_to_term(bin, [:safe])` rejects atoms the loading VM has
  not interned, and the manifest metadata carries `:exphil_policy` — an atom
  interned only by the EXPORTING module, which a pure-inference VM never
  loads.

  The gotcha's own closing line is why this test has to be shaped like this:

  > Round-trip tests in one VM prove NOTHING about fresh-VM loads when atoms
  > are involved. Tests could not catch it: the test file's own literals
  > intern the atom.

  So we spawn a peer node, ship it only the load call, and let it fail the
  way production failed. Tagged `:slow` — booting a node costs a second or
  two, which is still cheap next to three hours.
  """
  use ExUnit.Case, async: false

  @moduletag :slow

  # Any real exported policy. Chosen at runtime so the test does not rot when
  # checkpoints are pruned.
  defp some_checkpoint do
    "checkpoints/*.bin"
    |> Path.wildcard()
    |> Enum.reject(&String.contains?(&1, ["_latest", ".trainer.", "rejected"]))
    |> Enum.sort()
    |> List.first()
  end

  describe "checkpoint loads in a VM that never ran the exporter" do
    @tag timeout: 120_000
    test "a peer node can load a policy without interning the exporter's atoms" do
      checkpoint = some_checkpoint()

      if is_nil(checkpoint) do
        # No artifacts on this machine (fresh clone / CI): nothing to prove.
        :ok
      else
        # A peer node starts with a bare code path. We deliberately do NOT
        # preload ExPhil.Training.Imitation.Checkpoint (the exporter) there —
        # that omission IS the test.
        {:ok, peer, _node} = start_peer()

        result =
          :peer.call(peer, ExPhil.Training.Checkpoint, :load_policy, [checkpoint])

        :peer.stop(peer)

        assert match?({:ok, _}, result),
               """
               A fresh VM could not load #{checkpoint}: #{inspect(result)}

               This is GOTCHAS #68 — `binary_to_term(:safe)` rejecting atoms
               the loading VM never interned (the exporting module interns
               them; a pure-inference VM does not load it). A same-VM
               round-trip test cannot see this, which is why every probe
               crashed in production while the suite stayed green.

               Fix shape: trusted-artifact fallback to unrestricted
               binary_to_term, as in deserialize_trusted/2.
               """
      end
    end
  end

  defp start_peer do
    # Peer inherits this VM's code paths (so exphil modules are LOADABLE) but
    # loads nothing eagerly — atoms are interned only by what actually runs.
    # `connection: :standard_io` runs the peer WITHOUT Erlang distribution,
    # so this works in a plain (non-distributed) test VM. args must be
    # charlists — :peer.verify_args rejects binaries.
    :peer.start_link(%{
      name: :"fresh_load_#{System.unique_integer([:positive])}",
      connection: :standard_io,
      args: Enum.flat_map(:code.get_path(), fn p -> [~c"-pa", p] end),
      wait_boot: 30_000
    })
  end
end

defmodule ExPhil.Harness.FreshVmSafeAtomTest do
  @moduledoc """
  Negative control for `FreshVmLoadTest`: proves the peer-node setup can
  actually SEE the #68 failure, rather than passing because everything is
  loaded anyway.

  Without this, a green fresh-VM test proves nothing — exactly the trap the
  gotcha describes ("the test file's own literals intern the atom").
  """
  use ExUnit.Case, async: false

  @moduletag :slow

  @tag timeout: 120_000
  test "a :safe load of an un-interned atom DOES fail in a fresh VM" do
    # An atom no VM has any reason to intern.
    blob = :erlang.term_to_binary(%{arch: :"exphil_fresh_vm_probe_atom_xyz"})

    {:ok, peer, _node} =
      :peer.start_link(%{
        name: :"safe_probe_#{System.unique_integer([:positive])}",
        connection: :standard_io,
        wait_boot: 30_000
      })

    safe_result =
      try do
        :peer.call(peer, :erlang, :binary_to_term, [blob, [:safe]])
      catch
        kind, reason -> {kind, reason}
      end

    # The unrestricted load must succeed on the same bytes — that asymmetry
    # is the whole mechanism, and it is what deserialize_trusted/2 relies on.
    unrestricted = :peer.call(peer, :erlang, :binary_to_term, [blob])
    :peer.stop(peer)

    refute match?(%{arch: _}, safe_result),
           ":safe deserialization of an un-interned atom should FAIL in a fresh VM; " <>
             "if it succeeds, this harness cannot detect GOTCHA #68 and the sibling " <>
             "test's green result is meaningless."

    assert %{arch: :"exphil_fresh_vm_probe_atom_xyz"} = unrestricted
  end
end
