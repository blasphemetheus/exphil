defmodule ExPhil.Training.XlaExecCacheTest do
  # Pins for the cached-executable hang (JIT_WARMUP.md, bisected
  # 2026-08-24): deserializing the big `predict` executable defers
  # expensive finalization to FIRST USE in the calling process — the
  # Inference process's first live call lands mid-game, starves the
  # frame loop for seconds, and drops the session (local: spectator
  # disconnect; netplay: the both-peers freeze that convicted the
  # cache). The 3-arm bisect: all-cached FATAL, predict-only FATAL,
  # sampling-only CLEAN.
  #
  # Env-driven, so NOT async.
  use ExUnit.Case, async: false

  alias ExPhil.Training.Utils

  setup do
    prior = System.get_env("EXPHIL_XLA_EXEC_CACHE")
    prior_only = System.get_env("EXPHIL_XLA_EXEC_CACHE_ONLY")

    on_exit(fn ->
      restore = fn
        name, nil -> System.delete_env(name)
        name, v -> System.put_env(name, v)
      end

      restore.("EXPHIL_XLA_EXEC_CACHE", prior)
      restore.("EXPHIL_XLA_EXEC_CACHE_ONLY", prior_only)
    end)

    :ok
  end

  test "DEFAULT OFF: no cache opts for any function unless explicitly enabled" do
    System.delete_env("EXPHIL_XLA_EXEC_CACHE")

    for name <- ["predict", "trunk_step", "heads", "sampling_fused_det"] do
      assert Utils.xla_exec_cache(name, {:key}) == [],
             "#{name} got cache opts with the cache disabled"
    end

    System.put_env("EXPHIL_XLA_EXEC_CACHE", "0")
    assert Utils.xla_exec_cache("predict", {:key}) == []
  end

  test "EXPHIL_XLA_EXEC_CACHE_ONLY restricts caching per function (the bisect knob)" do
    System.put_env("EXPHIL_XLA_EXEC_CACHE", "1")
    System.put_env("EXPHIL_XLA_EXEC_CACHE_ONLY", "sampling")

    # The convicted function must get NO cache opts when excluded
    assert Utils.xla_exec_cache("predict", {:key}) == []
    assert Utils.xla_exec_cache("trunk_step", {:key}) == []

    # Prefix match covers every fused sampler
    assert [cache: path] = Utils.xla_exec_cache("sampling_fused_det", {:key})
    assert String.ends_with?(path, ".exlaexec")

    System.put_env("EXPHIL_XLA_EXEC_CACHE_ONLY", "predict,heads")
    assert [cache: _] = Utils.xla_exec_cache("predict", {:key})
    assert [cache: _] = Utils.xla_exec_cache("heads", {:key})
    assert Utils.xla_exec_cache("sampling_fused_det", {:key}) == []
  end

  test "enabled with no ONLY filter caches everything, keyed by name+parts" do
    System.put_env("EXPHIL_XLA_EXEC_CACHE", "1")
    System.delete_env("EXPHIL_XLA_EXEC_CACHE_ONLY")

    [cache: a] = Utils.xla_exec_cache("predict", {:k1})
    [cache: b] = Utils.xla_exec_cache("predict", {:k2})
    [cache: c] = Utils.xla_exec_cache("heads", {:k1})
    assert a != b and a != c
  end
end
