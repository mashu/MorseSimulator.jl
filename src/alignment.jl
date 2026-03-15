"""
    alignment.jl — Expose simulator timings as token–frame ranges.

The simulator already has full timings (TimedMorseEvent per element; transmission_ranges).
This module does not "align" or estimate; it reports which tokens (letters/specials)
belong to which time/frame ranges by converting the simulator's times to frame indices.
Downstream (e.g. MorseDecoder) uses TokenTiming to slice chunks so each chunk's
labels match the spectrogram. Token order matches decoder tokenization.
"""

"""
    AbstractTokenTiming

Supertype for token-to-frame alignment. Use `TokenTiming` when alignment is available,
`NoTiming` when it is not (e.g. alignment failed or not computed). Enables dispatch
without Union or conditionals.
"""
abstract type AbstractTokenTiming end

"""
    TokenTiming

Per-token frame ranges for the label. Same length as the token sequence
(label_to_token_ids(label) in decoder terms).

# Fields
- `token_start_frames::Vector{Int}` — 1-based start frame index per token
- `token_end_frames::Vector{Int}` — 1-based end frame index per token (inclusive)
"""
struct TokenTiming <: AbstractTokenTiming
    token_start_frames::Vector{Int}
    token_end_frames::Vector{Int}
end

"""No token timing available; use proportional mapping downstream."""
struct NoTiming <: AbstractTokenTiming end

# Match decoder special token strings so tokenization aligns (no leading _ per style)
const LABEL_SPECIAL_START = "<START>"
const LABEL_SPECIAL_END   = "<END>"
const LABEL_SPECIAL_TS    = "[TS]"
const LABEL_SPECIAL_TE    = "[TE]"
const LABEL_SPEAKER_TAGS  = ["[S1]", "[S2]", "[S3]", "[S4]", "[S5]", "[S6]"]
const LABEL_SPECIALS      = Set(String[LABEL_SPECIAL_START, LABEL_SPECIAL_END, LABEL_SPECIAL_TS, LABEL_SPECIAL_TE, LABEL_SPEAKER_TAGS...])

"""
    character_time_ranges(text, events) -> Vector{Tuple{Float64,Float64}}

Return (start_time, end_time) in seconds for each character (and space between words)
in `text`, using the timed events for that transmission. Events must be in the same
order as produced by text_to_timed_events for this text. Used to align label tokens
to spectrogram frames.
"""
function character_time_ranges(text::AbstractString, events::Vector{TimedMorseEvent{Float64}})
    out = Tuple{Float64, Float64}[]
    isempty(events) && return out
    ev_idx = 1
    words = split(strip(text))
    for (wi, word) in enumerate(words)
        word_str = String(word)
        symbols_list = if is_prosign(word_str)
            [prosign_to_morse(word_str)]
        else
            [char_to_morse(c) for c in word_str]
        end
        for (ci, symbols) in enumerate(symbols_list)
            isempty(symbols) && continue
            n_sym = length(symbols)
            # Events for this character: n_sym elements + (n_sym - 1) symbol gaps
            n_ev = 2 * n_sym - 1
            # Char gap after (unless last char in word)
            if ci < length(symbols_list)
                n_ev += 1
            end
            ev_end = ev_idx + n_ev - 1
            ev_end > length(events) && break
            t_start = events[ev_idx].start_time
            t_end = events[ev_end].start_time + events[ev_end].duration
            push!(out, (t_start, t_end))
            ev_idx = ev_end + 1
        end
        # Word gap (one event) between words → we emit a (start,end) for the space token
        if wi < length(words)
            ev_idx > length(events) && break
            t_start = events[ev_idx].start_time
            t_end = t_start + events[ev_idx].duration
            push!(out, (t_start, t_end))
            ev_idx += 1
        end
    end
    return out
end

"""
    compute_token_timings(transcript, scene_events) -> Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}}

Return (start_times, end_times) in seconds, one entry per token in label order, or nothing
if alignment cannot be computed. Does not throw.
"""
function compute_token_timings(
    transcript::Transcript,
    scene_events::SceneMorseEvents{Float64},
)::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}}
    sorted_txs = sort(collect(transcript.transmissions), by = tx -> tx.time_offset)
    length(scene_events.transmission_ranges) != length(sorted_txs) && return nothing
    station_events = Dict(se.callsign => se.events for se in scene_events.station_events)
    total_dur = scene_events.total_duration

    # Build token list and times in same order as decoder (split label, expand)
    label = transcript.label
    parts = split(label)
    start_times = Float64[]
    end_times = Float64[]

    first_tx_start = isempty(sorted_txs) ? total_dur : sorted_txs[1].time_offset
    tx_idx = 0
    char_ranges_idx = 0
    char_ranges = Tuple{Float64, Float64}[]
    current_tx_start = 0.0
    current_tx_end = total_dur

    for (i, part) in enumerate(parts)
        if part == LABEL_SPECIAL_START
            push!(start_times, 0.0)
            push!(end_times, first_tx_start)
            continue
        end
        if part == LABEL_SPECIAL_END
            push!(start_times, current_tx_end)
            push!(end_times, total_dur)
            continue
        end
        if part == LABEL_SPECIAL_TS
            tx_idx += 1
            if tx_idx <= length(sorted_txs) && tx_idx <= length(scene_events.transmission_ranges)
                tx = sorted_txs[tx_idx]
                (call, s, e) = scene_events.transmission_ranges[tx_idx]
                evs = get(station_events, call, nothing)
                evs === nothing && return nothing
                s < 1 || e > length(evs) && return nothing
                current_tx_start = evs[s].start_time
                ev = evs[e]
                current_tx_end = ev.start_time + ev.duration
                seg_evs = evs[s:e]
                char_ranges = character_time_ranges(tx.text, seg_evs)
                char_ranges_idx = 0
                push!(start_times, current_tx_start)
                push!(end_times, current_tx_start)
            else
                push!(start_times, total_dur)
                push!(end_times, total_dur)
            end
            continue
        end
        if part in LABEL_SPEAKER_TAGS
            push!(start_times, current_tx_start)
            push!(end_times, current_tx_start)
            continue
        end
        if part == LABEL_SPECIAL_TE
            push!(start_times, current_tx_end)
            push!(end_times, current_tx_end)
            continue
        end
        # Part is text (word or words): one token per character, same order as character_time_ranges
        for (ci, c) in enumerate(part)
            char_ranges_idx += 1
            if char_ranges_idx <= length(char_ranges)
                t_start, t_end = char_ranges[char_ranges_idx]
                push!(start_times, t_start)
                push!(end_times, t_end)
            else
                push!(start_times, total_dur)
                push!(end_times, total_dur)
            end
        end
        # Space between parts (decoder adds space if next part is not special)
        if i < length(parts) && !(parts[i + 1] in LABEL_SPECIALS)
            if char_ranges_idx < length(char_ranges)
                char_ranges_idx += 1
                t_start, t_end = char_ranges[char_ranges_idx]
                push!(start_times, t_start)
                push!(end_times, t_end)
            else
                push!(start_times, total_dur)
                push!(end_times, total_dur)
            end
        end
    end

    (start_times, end_times)
end

"""
    time_to_frame(t_sec, n_frames, total_duration_sec, sample_rate, hop_size) -> Int

Map time in seconds to 1-based frame index. Frame j covers time
(j-1)*hop_size/sample_rate. Clamped to 1:n_frames.
"""
function time_to_frame(
    t_sec::Float64,
    n_frames::Int,
    total_duration_sec::Float64,
    sample_rate::Int,
    hop_size::Int,
)::Int
    # Frame center times: (j-1) * hop_size / sample_rate for j in 1:n_frames
    frame_idx = round(Int, t_sec * sample_rate / hop_size) + 1
    clamp(frame_idx, 1, n_frames)
end

"""
    token_timings_to_frames(start_times, end_times, n_frames, total_duration_sec, sample_rate, hop_size)
        -> (start_frames::Vector{Int}, end_frames::Vector{Int})

Convert token time ranges to 1-based frame indices.
"""
function token_timings_to_frames(
    start_times::Vector{Float64},
    end_times::Vector{Float64},
    n_frames::Int,
    total_duration_sec::Float64,
    sample_rate::Int,
    hop_size::Int,
)
    start_frames = Int[]
    end_frames = Int[]
    for i in 1:length(start_times)
        push!(start_frames, time_to_frame(start_times[i], n_frames, total_duration_sec, sample_rate, hop_size))
        push!(end_frames, time_to_frame(end_times[i], n_frames, total_duration_sec, sample_rate, hop_size))
    end
    (start_frames, end_frames)
end

"""
    compute_token_timing(transcript, scene_events, n_frames, total_duration_sec, sample_rate, hop_size)
        -> AbstractTokenTiming

Returns TokenTiming when alignment succeeds (lengths consistent), NoTiming() otherwise.
Does not throw; use for dataset generation without try-catch.
"""
function compute_token_timing(
    transcript::Transcript,
    scene_events::SceneMorseEvents{Float64},
    n_frames::Int,
    total_duration_sec::Float64,
    sample_rate::Int,
    hop_size::Int,
)::AbstractTokenTiming
    timings = compute_token_timings(transcript, scene_events)
    timings === nothing && return NoTiming()
    start_times, end_times = timings
    L = length(start_times)
    (length(end_times) != L || L == 0) && return NoTiming()
    start_f, end_f = token_timings_to_frames(
        start_times, end_times, n_frames, total_duration_sec, sample_rate, hop_size,
    )
    (length(start_f) != L || length(end_f) != L) && return NoTiming()
    TokenTiming(start_f, end_f)
end
