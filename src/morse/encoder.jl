"""
    MorseSimulator.jl/src/morse/encoder.jl

Encoder converting transcripts to sequences of timed Morse events
for each station, with proper inter-transmission gaps and
operator pauses.
"""

using Random, Distributions

"""
    StationMorseEvents{T<:AbstractFloat}

Timed Morse events for a single station.

# Fields
- `callsign::String` — station callsign
- `events::Vector{TimedMorseEvent{T}}` — timed Morse events
- `tone_freq::Float64` — CW tone frequency (Hz)
- `signal_amplitude::Float64` — signal amplitude (linear)
"""
struct StationMorseEvents{T<:AbstractFloat}
    callsign::String
    events::Vector{TimedMorseEvent{T}}
    tone_freq::Float64
    signal_amplitude::Float64
end

"""
    SceneMorseEvents{T<:AbstractFloat}

All Morse events for a complete band scene.

# Fields
- `station_events::Vector{StationMorseEvents{T}}`
- `total_duration::T`
- `label::String`
"""
struct SceneMorseEvents{T<:AbstractFloat}
    station_events::Vector{StationMorseEvents{T}}
    total_duration::T
    label::String
end

"""
    encode_transcript(rng, transcript, scene) -> SceneMorseEvents

Convert a transcript into timed Morse events for all stations.
"""
function encode_transcript(rng::AbstractRNG, transcript::Transcript, scene::BandScene)
    # Group transmissions by callsign, build station lookup
    station_lookup = Dict(s.callsign => s for s in scene.stations)

    station_events_map = Dict{String, Vector{TimedMorseEvent{Float64}}}()

    for tx in transcript.transmissions
        station = get(station_lookup, tx.callsign, nothing)
        station === nothing && continue

        events = text_to_timed_events(
            rng, tx.text, tx.wpm;
            start_time = tx.time_offset,
            jitter = 0.05,
            drift = 0.01
        )

        if haskey(station_events_map, tx.callsign)
            append!(station_events_map[tx.callsign], events)
        else
            station_events_map[tx.callsign] = events
        end
    end

    # Build StationMorseEvents (use received amplitude with path loss so both audio and direct path match)
    all_station_events = StationMorseEvents{Float64}[]
    for (call, events) in station_events_map
        station = station_lookup[call]
        received_amplitude = signal_strength(rng, scene, station)
        push!(all_station_events, StationMorseEvents{Float64}(
            call, events, station.tone_freq, received_amplitude
        ))
    end

    # Compute total duration
    max_dur = 0.0
    for se in all_station_events
        d = total_duration(se.events)
        d > max_dur && (max_dur = d)
    end
    max_dur += 0.5  # Add trailing silence

    return SceneMorseEvents{Float64}(all_station_events, max_dur, transcript.label)
end

encode_transcript(transcript::Transcript, scene::BandScene) =
    encode_transcript(Random.default_rng(), transcript, scene)
