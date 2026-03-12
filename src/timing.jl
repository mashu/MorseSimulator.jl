"""
    CWContestSim/src/morse/timing.jl

Morse timing model converting WPM to element durations with
realistic operator jitter and speed variation.
"""

using Random, Distributions

# ============================================================================
# Timing Parameters
# ============================================================================

"""
    TimingParams{T<:AbstractFloat}

Timing parameters for Morse code generation.

# Fields
- `dot_duration::T` — base dot duration in seconds
- `jitter_sigma::T` — timing jitter std dev (fraction of dot_duration)
- `speed_drift_rate::T` — how quickly WPM drifts during transmission
"""
struct TimingParams{T<:AbstractFloat}
    dot_duration::T
    jitter_sigma::T
    speed_drift_rate::T
end

"""
    TimingParams(wpm; jitter, drift) -> TimingParams{Float64}

Create timing parameters from words-per-minute.
PARIS standard: 1 WPM = 50 dot units per minute.
"""
function TimingParams(wpm::Real;
                      jitter::Float64 = 0.05,
                      drift::Float64 = 0.01)
    dot_dur = 1.2 / Float64(wpm)  # 60 / (50 * wpm)
    TimingParams{Float64}(dot_dur, jitter, drift)
end

# ============================================================================
# Timed Morse Event
# ============================================================================

"""
    TimedMorseEvent{T<:AbstractFloat}

A Morse element with absolute timing information.

# Fields
- `element::AbstractMorseElement` — the Morse element
- `start_time::T` — start time in seconds
- `duration::T` — actual duration in seconds (with jitter)
"""
struct TimedMorseEvent{T<:AbstractFloat}
    element::AbstractMorseElement
    start_time::T
    duration::T
end

# ============================================================================
# Duration Computation with Jitter
# ============================================================================

"""
    jittered_duration(rng, element, params) -> Float64

Compute the actual duration of a Morse element with timing jitter.
"""
function jittered_duration(rng::AbstractRNG, element::AbstractMorseElement,
                           params::TimingParams{T}) where T
    base = dot_units(element) * params.dot_duration
    jitter = randn(rng) * params.jitter_sigma * params.dot_duration
    return max(base * T(0.3), base + jitter)  # Never less than 30% of base
end

# ============================================================================
# Text to Timed Events
# ============================================================================

"""
    text_to_timed_events(rng, text, wpm; start_time, jitter, drift) -> Vector{TimedMorseEvent}

Convert text string to a sequence of timed Morse events.
WPM may drift during the transmission.
"""
function text_to_timed_events(rng::AbstractRNG, text::AbstractString, wpm::Float64;
                               start_time::Float64 = 0.0,
                               jitter::Float64 = 0.05,
                               drift::Float64 = 0.01)
    events = TimedMorseEvent{Float64}[]
    current_wpm = wpm
    t = start_time
    words = split(strip(text))

    for (wi, word) in enumerate(words)
        word_str = String(word)

        # Check if word is a prosign
        symbols_list = if is_prosign(word_str)
            [prosign_to_morse(word_str)]  # Single character group
        else
            [char_to_morse(c) for c in word_str]
        end

        for (ci, symbols) in enumerate(symbols_list)
            isempty(symbols) && continue

            # WPM drift
            current_wpm = max(5.0, current_wpm + randn(rng) * drift * wpm)
            params = TimingParams(current_wpm; jitter=jitter, drift=drift)

            for (si, sym) in enumerate(symbols)
                dur = jittered_duration(rng, sym, params)
                push!(events, TimedMorseEvent{Float64}(sym, t, dur))
                t += dur

                # Symbol gap (between dots/dashes within character)
                if si < length(symbols)
                    gap_dur = jittered_duration(rng, SymbolGap(), params)
                    push!(events, TimedMorseEvent{Float64}(SymbolGap(), t, gap_dur))
                    t += gap_dur
                end
            end

            # Character gap (between characters within word, but not for prosigns which are one unit)
            if ci < length(symbols_list)
                params_gap = TimingParams(current_wpm; jitter=jitter, drift=drift)
                gap_dur = jittered_duration(rng, CharGap(), params_gap)
                push!(events, TimedMorseEvent{Float64}(CharGap(), t, gap_dur))
                t += gap_dur
            end
        end

        # Word gap
        if wi < length(words)
            params_word = TimingParams(current_wpm; jitter=jitter, drift=drift)
            gap_dur = jittered_duration(rng, WordGap(), params_word)
            push!(events, TimedMorseEvent{Float64}(WordGap(), t, gap_dur))
            t += gap_dur
        end
    end

    return events
end

"""
    total_duration(events) -> Float64

Total duration of a sequence of timed events.
"""
function total_duration(events::Vector{TimedMorseEvent{T}}) where T
    isempty(events) && return zero(T)
    last_ev = events[end]
    return last_ev.start_time + last_ev.duration
end
