"""
    MorseSimulator.jl/src/transcript/transmission.jl

Transmission record type representing a single over-the-air
transmission by one station. Forms the building blocks of
transcripts.
"""

"""
    Transmission

A single CW transmission by one station.

# Fields
- `callsign::String` — transmitting station's callsign
- `text::String` — the CW text content
- `time_offset::Float64` — start time in seconds from scene start
- `duration_estimate::Float64` — estimated duration in seconds
- `wpm::Float64` — sending speed for this transmission
- `signal_strength::Float64` — received signal strength (linear)
"""
struct Transmission
    callsign::String
    text::String
    time_offset::Float64
    duration_estimate::Float64
    wpm::Float64
    signal_strength::Float64
end

"""
    Transcript

A complete transcript of a simulated CW scene.

# Fields
- `transmissions::Vector{Transmission}` — ordered list of transmissions
- `label::String` — plain text label for ML training
- `mode_name::String` — conversation mode name
- `contest_name::String` — contest type name
"""
struct Transcript
    transmissions::Vector{Transmission}
    label::String
    mode_name::String
    contest_name::String
end

"""
    flat_text(transcript) -> String

Produce the flat text label: <START> ... <END>
"""
function flat_text(t::Transcript)
    return t.label
end

"""
    annotated_text(transcript) -> String

Produce annotated text with TX markers for inspection.
"""
function annotated_text(t::Transcript)
    parts = String[]
    push!(parts, "<START>")
    for tx in t.transmissions
        push!(parts, "<TX_START:$(tx.callsign)>")
        push!(parts, tx.text)
        push!(parts, "<TX_END>")
    end
    push!(parts, "<END>")
    return join(parts, " ")
end

"""
    estimate_duration(text, wpm) -> Float64

Estimate transmission duration in seconds from text and WPM.
Uses PARIS standard: 1 WPM ≈ 50 dot units per minute.
Average word length ≈ 5 characters.
"""
function estimate_duration(text::AbstractString, wpm::Float64)
    # Each character averages ~10 dot units, word gap = 7 dots
    # At given WPM, dot_duration = 1.2 / wpm seconds
    dot_dur = 1.2 / wpm
    nchars = count(!isspace, text)
    nspaces = count(isspace, text)
    # Rough: chars * 10 dot units + spaces * 7 dot units
    total_dots = nchars * 10.0 + nspaces * 7.0
    return total_dots * dot_dur
end
