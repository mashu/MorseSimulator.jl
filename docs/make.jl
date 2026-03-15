using Documenter, MorseSimulator

makedocs(
    sitename = "MorseSimulator.jl",
    modules = [MorseSimulator],
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://mashu.github.io/MorseSimulator.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Architecture" => "architecture.md",
        "Transcripts" => "transcripts.md",
        "Signal & Spectrogram" => "signal.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/mashu/MorseSimulator.jl.git",
    devbranch = "main",
    push_preview = true,
)
