logits = Nx.tensor([[-2.0, 0.5, 3.0, -70.0, 65.0, 1.0, -1.0, 0.0]], type: :f32, backend: Nx.BinaryBackend)
targets = Nx.tensor([[0, 1, 0, 1, 0, 1, 0, 0]], type: :f32, backend: Nx.BinaryBackend)

mk_loss = fn clip? ->
  fn lg ->
    lg = if clip?, do: Nx.clip(lg, -60.0, 60.0), else: lg
    max_val = Nx.max(lg, 0)
    abs_l = Nx.abs(lg)
    loss = Nx.subtract(max_val, Nx.multiply(lg, targets))
    loss = Nx.add(loss, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_l)))))
    Nx.mean(loss)
  end
end

# analytic reference: (sigmoid(clamped_x) - t) / 8 with 0 outside clamp
run = fn label, clip?, compiler ->
  opts = if compiler, do: [compiler: compiler], else: []
  {_l, g} = Nx.Defn.jit_apply(fn l -> Nx.Defn.value_and_grad(l, mk_loss.(clip?)) end, [logits], opts)
  IO.puts("#{label}: #{inspect(Enum.map(Nx.to_flat_list(g), &Float.round(&1 * 1.0, 5)))}")
end

expected = [0.01492, -0.04689, 0.11928, 0.0, 0.0, -0.03356, 0.03356, 0.0625]
IO.puts("expected (clip): #{inspect(expected)}")
run.("clip+EXLA     ", true, EXLA)
run.("clip+evaluator", true, nil)
run.("raw +EXLA     ", false, EXLA)
run.("raw +evaluator", false, nil)

mk_loss_mm = fn lg ->
  lg = Nx.min(Nx.max(lg, -60.0), 60.0)
  max_val = Nx.max(lg, 0)
  abs_l = Nx.abs(lg)
  loss = Nx.subtract(max_val, Nx.multiply(lg, targets))
  loss = Nx.add(loss, Nx.log(Nx.add(1, Nx.exp(Nx.negate(abs_l)))))
  Nx.mean(loss)
end

{_l, g} = Nx.Defn.jit_apply(fn l -> Nx.Defn.value_and_grad(l, mk_loss_mm) end, [logits], compiler: EXLA)
IO.puts("minmax+EXLA   : #{inspect(Enum.map(Nx.to_flat_list(g), &Float.round(&1 * 1.0, 5)))}")
