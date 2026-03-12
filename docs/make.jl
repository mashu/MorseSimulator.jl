using Documenter, MorseSimulator

makedocs(
    sitename = "MorseSimulator.jl",
    modules = [MorseSimulator],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Home" => "index.md",
        "Architecture" => "architecture.md",
        "Phase 1: Transcripts" => "transcripts.md",
        "Phase 2: Signal & Spectrogram" => "signal.md",
        "API Reference" => "api.md",
    ]
)
