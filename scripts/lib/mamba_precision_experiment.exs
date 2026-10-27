defmodule MambaPrecisionExperiment do
  @moduledoc "Diagnostic-only primitive replacement inside this VM; source files are never modified."
  def install(mode) when is_binary(mode) do
    case String.split(mode, "+") do
      [base] ->
        install_base(base)

      [base | stages] ->
        install_base(base)
        install_stages(stages)
    end
  end

  defp install_base("baseline"), do: :ok

  defp install_base(mode)
       when mode in ["dense", "norm", "input", "row_dot", "chunk32", "chunk8", "dense_norm"] do
    path = Path.expand("../edifice/lib/edifice/stateful/ops.ex")
    source = File.read!(path)

    source =
      if mode == "dense_norm" do
        source
        |> String.replace("def layer_norm(x,", "def original_layer_norm(x,")
        |> String.replace("defmodule Edifice.Stateful.Ops do", """
        defmodule Edifice.Stateful.Ops do
          def layer_norm(x, params, epsilon \\\\ 1.0e-5) do
            type = Nx.type(x)
            p = Nx.Defn.Composite.traverse(params, &Nx.as_type(&1, :f64))
            original_layer_norm(Nx.as_type(x, :f64), p, epsilon) |> Nx.as_type(type)
          end
        """)
      else
        source
      end

    wrapper =
      if mode == "norm" do
        source = String.replace(source, "def layer_norm(x,", "def original_layer_norm(x,")

        {source,
         """
         def layer_norm(x, params, epsilon \\\\ 1.0e-5) do
           type = Nx.type(x)
           p = Nx.Defn.Composite.traverse(params, &Nx.as_type(&1, :f64))
           original_layer_norm(Nx.as_type(x, :f64), p, epsilon) |> Nx.as_type(type)
         end
         """}
      else
        source = String.replace(source, "def dense(x,", "def original_dense(x,")

        body =
          if mode in ["chunk32", "chunk8"] do
            chunk = if mode == "chunk32", do: 32, else: 8

            """
            if Nx.type(x) == {:f,64} do
              original_dense(x, params)
            else
              kernel = params["kernel"]
              width = Nx.axis_size(x,1)
              chunk = #{chunk}
              groups = div(width + chunk - 1, chunk)
              padding = groups * chunk - width
              x = Nx.pad(x, 0.0, [{0,0,0},{0,padding,0}])
              kernel = Nx.pad(kernel, 0.0, [{0,padding,0},{0,0,0}])
              x = Nx.reshape(x, {Nx.axis_size(x,0),groups,chunk}) |> Nx.transpose(axes: [1,0,2])
              kernel = Nx.reshape(kernel, {groups,chunk,Nx.axis_size(kernel,1)})
              product = Nx.dot(x,[2],[0],kernel,[1],[0]) |> Nx.as_type(:f64) |> Nx.sum(axes: [0])
              product = if params["bias"], do: Nx.add(product,Nx.as_type(params["bias"],:f64)), else: product
              Nx.as_type(product,:f32)
            end
            """
          else
            if mode == "row_dot" do
              """
              n = Nx.axis_size(x, 0)
              k = params["kernel"]
              weights = Nx.broadcast(Nx.new_axis(k, 0), {n, Nx.axis_size(k,0), Nx.axis_size(k,1)})
              product = Nx.dot(Nx.new_axis(x,1), [2], [0], weights, [1], [0]) |> Nx.squeeze(axes: [1])
              if params["bias"], do: Nx.add(product, params["bias"]), else: product
              """
            else
              """
              type = Nx.type(x)
              p = Nx.Defn.Composite.traverse(params, &Nx.as_type(&1, :f64))
              original_dense(Nx.as_type(x, :f64), p) |> Nx.as_type(type)
              """
            end
          end

        body =
          if mode == "input",
            do: "if Nx.axis_size(x, 1) == 264 do\n#{body}\nelse\noriginal_dense(x, params)\nend",
            else: body

        {source, "def dense(x, params) do\n#{body}\nend\n"}
      end

    {source, wrapper} = wrapper

    source =
      String.replace(
        source,
        "defmodule Edifice.Stateful.Ops do",
        "defmodule Edifice.Stateful.Ops do\n" <> wrapper,
        global: false
      )

    Code.compile_string(source, path <> ".experiment")
  end

  defp install_stages(stages) do
    unless Enum.all?(stages, &(&1 in ["ssm", "update", "readout", "gate", "output"])) and
             length(Enum.uniq(stages)) == length(stages) and
             Enum.count(stages, &(&1 in ["ssm", "update", "readout"])) <= 1 and
             Enum.count(stages, &(&1 in ["gate", "output"])) <= 1,
           do: raise(ArgumentError, "Choose one SSM stage and/or one output stage")

    path = Path.expand("../edifice/lib/edifice/ssm/common.ex")
    source = File.read!(path)
    source = Enum.reduce(stages, source, &replace_stage/2)
    Code.compile_string(source, path <> ".experiment")
  end

  defp replace_stage(stage, source) when stage in ["ssm", "update"] do
    body =
      if stage == "ssm" do
        "{Nx.as_type(y, Nx.type(x)), Nx.as_type(next_h, Nx.type(h))}"
      else
        """
        next_h = Nx.as_type(next_h, Nx.type(h))
        y = Nx.sum(Nx.multiply(Nx.new_axis(c, 1), next_h), axes: [2])
        {y, next_h}
        """
      end

    source
    |> replace_once!(
      "def step_discretized_ssm(x, b, c, dt, h) do",
      "def original_step_discretized_ssm(x, b, c, dt, h) do"
    )
    |> replace_once!("defmodule Edifice.SSM.Common do", """
    defmodule Edifice.SSM.Common do
      def step_discretized_ssm(x, b, c, dt, h) do
        {#{if stage == "update", do: "_y", else: "y"}, next_h} = original_step_discretized_ssm(
          Nx.as_type(x, :f64), Nx.as_type(b, :f64), Nx.as_type(c, :f64),
          Nx.as_type(dt, :f64), Nx.as_type(h, :f64))
        #{body}
      end
    """)
  end

  defp replace_stage("readout", source) do
    replace_once!(
      source,
      "y = Nx.sum(Nx.multiply(Nx.new_axis(c, 1), new_h), axes: [2])",
      "y = Nx.sum(Nx.multiply(Nx.new_axis(Nx.as_type(c, :f64), 1), Nx.as_type(new_h, :f64)), axes: [2]) |> Nx.as_type(Nx.type(x))"
    )
  end

  defp replace_stage(stage, source) when stage in ["gate", "output"] do
    source =
      replace_once!(
        source,
        "gated = Nx.multiply(y, Ops.silu(z_branch))",
        "gated = Nx.multiply(Nx.as_type(y, :f64), Ops.silu(Nx.as_type(z_branch, :f64)))" <>
          if(stage == "gate", do: " |> Nx.as_type(Nx.type(y))", else: "")
      )

    if stage == "output" do
      replace_once!(
        source,
        ~S'out = Ops.dense(gated, Ops.layer_params!(params, "#{name}_out_proj"))',
        ~S'out_params = Ops.layer_params!(params, "#{name}_out_proj") |> Nx.Defn.Composite.traverse(&Nx.as_type(&1, :f64))' <>
          "\nout = Ops.dense(gated, out_params) |> Nx.as_type(Nx.type(frame))"
      )
    else
      source
    end
  end

  defp replace_once!(source, old, new) do
    unless length(String.split(source, old)) == 2,
      do: raise(ArgumentError, "Expected exactly one source match for #{inspect(old)}")

    String.replace(source, old, new, global: false)
  end
end
