using Documenter, CWContestSim

makedocs(
    sitename = "CWContestSim.jl",
    modules = [CWContestSim],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Home" => "index.md",
        "Architecture" => "architecture.md",
        "Phase 1: Transcripts" => "transcripts.md",
        "Phase 2: Signal & Spectrogram" => "signal.md",
        "API Reference" => "api.md",
    ]
)
