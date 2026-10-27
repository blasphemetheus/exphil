defmodule MambaPrecisionExperiment do
  @moduledoc "Diagnostic-only primitive replacement inside this VM; source files are never modified."
  def install("baseline"), do: :ok

  def install(mode) when mode in ["dense", "norm", "input", "row_dot", "chunk32", "chunk8"] do
    path = Path.expand("../edifice/lib/edifice/stateful/ops.ex")
    source = File.read!(path)

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
end
