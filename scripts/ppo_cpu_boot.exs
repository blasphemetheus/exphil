# Load existing modules on the CPU only; intentionally does not invoke Mix.
Application.put_env(:exla, :clients, host: [platform: :host])
Application.put_env(:exla, :default_client, :host)
Application.put_env(:nx, :default_backend, EXLA.Backend)
Application.put_env(:nx, :default_defn_options, compiler: EXLA)
{:ok, _} = Application.ensure_all_started(:exphil)
