defmodule ExPhil.Training.Callbacks.PolicyExportFailureTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{Config, Imitation, TrainingState}
  alias ExPhil.Training.Callbacks.PolicyExport

  @tag :tmp_dir
  test "required policy and config failures cannot report successful completion", %{tmp_dir: dir} do
    trainer = Imitation.new(embed_size: 16, hidden_sizes: [8], temporal: false)
    path = Path.join(dir, "model.axon")
    policy_path = Config.derive_policy_path(path)
    config_path = Config.derive_config_path(path)
    state = %TrainingState{trainer: trainer, opts: []}
    callback = PolicyExport.init(checkpoint_path: path)
    File.mkdir_p!(policy_path)

    assert_raise RuntimeError, ~r/Policy export failed/, fn ->
      PolicyExport.on_train_end(state, callback)
    end

    File.rmdir!(policy_path)
    File.mkdir_p!(config_path)
    assert_raise File.Error, fn -> PolicyExport.on_train_end(state, callback) end
  end
end
