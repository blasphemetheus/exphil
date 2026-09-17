base = Path.join(Path.dirname(__ENV__.file), "verified_sources")
File.mkdir_p!(base)
hash = fn bytes -> :crypto.hash(:sha256, bytes) |> Base.encode16(case: :lower) end

sources =
  for name <- ["exphil", "edifice", "nx", "libmelee_ex"] do
    repo = Path.expand("../#{name}")
    {head, 0} = System.cmd("git", ["-C", repo, "rev-parse", "HEAD"])
    {diff, 0} = System.cmd("git", ["-C", repo, "diff", "HEAD", "--binary"])
    {status, 0} = System.cmd("git", ["-C", repo, "status", "--short"])

    {paths, 0} =
      System.cmd("git", [
        "-C",
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z"
      ])

    files =
      String.split(paths, <<0>>, trim: true)
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.filter(fn p ->
        not String.starts_with?(p, "eval_runs/") and
          Path.extname(p) in [".ex", ".exs", ".rs", ".toml", ".lock", ".nix", ".c", ".h", ".cc"] and
          File.regular?(Path.join(repo, p))
      end)

    rows = Enum.map(files, fn p -> %{path: p, sha256: hash.(File.read!(Path.join(repo, p)))} end)
    File.write!(Path.join(base, name <> ".patch"), diff)
    File.write!(Path.join(base, name <> ".status"), status)
    list_path = Path.join(base, name <> ".files")
    File.write!(list_path, Enum.join(files, <<0>>) <> <<0>>)

    {_, 0} =
      System.cmd("tar", [
        "-czf",
        Path.join(base, name <> "_source.tar.gz"),
        "-C",
        repo,
        "--null",
        "-T",
        list_path
      ])

    %{repository: repo, head: String.trim(head), tracked_diff_sha256: hash.(diff), files: rows}
  end

report = %{
  captured_at: DateTime.utc_now() |> DateTime.to_iso8601(),
  sources: sources,
  note:
    "Includes pre-existing shared work; no commits or pushes. Source archives include listed code/config files. Evaluation scripts and manifests remain beside this report."
}

File.write!(Path.join(base, "provenance.json"), JSON.encode!(report))
