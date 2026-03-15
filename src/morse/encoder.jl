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
- `transmission_ranges::Vector{Tuple{String, Int, Int}}` — for each transmission in label order
  (time-sorted): (callsign, start_event_idx, end_event_idx) into that station's events.
  Used for token-to-frame alignment so chunks get the correct letters.
"""
struct SceneMorseEvents{T<:AbstractFloat}
    station_events::Vector{StationMorseEvents{T}}
    total_duration::T
    label::String
    transmission_ranges::Vector{Tuple{String, Int, Int}}
end

"""
    encode_transcript(rng, transcript, scene) -> SceneMorseEvents

Convert a transcript into timed Morse events for all stations.
Processes transmissions in time order so transmission_ranges match label order.
"""
function encode_transcript(rng::AbstractRNG, transcript::Transcript, scene::BandScene)
    station_lookup = Dict(s.callsign => s for s in scene.stations)
    station_events_map = Dict{String, Vector{TimedMorseEvent{Float64}}}()
    # Per-station current length so we can record (start, end) for each transmission
    station_len = Dict{String, Int}()

    # Process in label order (time order) so transmission_ranges align with label tokens
    sorted_txs = sort(collect(transcript.transmissions), by = tx -> tx.time_offset)
    transmission_ranges = Tuple{String, Int, Int}[]

    for tx in sorted_txs
        station = get(station_lookup, tx.callsign, nothing)
        station === nothing && continue

        events = text_to_timed_events(
            rng, tx.text, tx.wpm;
            start_time = tx.time_offset,
            jitter = 0.05,
            drift = 0.01
        )
        n = length(events)
        n == 0 && continue

        start_idx = get(station_len, tx.callsign, 0) + 1
        end_idx = start_idx + n - 1
        push!(transmission_ranges, (tx.callsign, start_idx, end_idx))
        station_len[tx.callsign] = end_idx

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

    max_dur = 0.0
    for se in all_station_events
        d = total_duration(se.events)
        d > max_dur && (max_dur = d)
    end
    max_dur += 0.5  # Add trailing silence

    return SceneMorseEvents{Float64}(all_station_events, max_dur, transcript.label, transmission_ranges)
end

encode_transcript(transcript::Transcript, scene::BandScene) =
    encode_transcript(Random.default_rng(), transcript, scene)
