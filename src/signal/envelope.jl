"""
    MorseSimulator.jl/src/signal/envelope.jl

Morse envelope generation. Creates the amplitude envelope
for CW tone signals from timed Morse events.
Includes rise/fall time shaping to avoid key clicks.
"""

"""
    RaisedCosineEnvelope{T} <: AbstractEnvelope{T}

Raised-cosine shaped envelope for smooth key transitions.

# Fields
- `rise_time::T` — rise/fall time in seconds (typically 5ms)
"""
struct RaisedCosineEnvelope{T<:AbstractFloat} <: AbstractEnvelope{T}
    rise_time::T
end

RaisedCosineEnvelope() = RaisedCosineEnvelope{Float64}(0.005)

"""
    HardEnvelope{T} <: AbstractEnvelope{T}

Hard (square) keying envelope. Sharp transitions, produces key clicks.
Useful for simulating poorly adjusted transmitters.
"""
struct HardEnvelope{T<:AbstractFloat} <: AbstractEnvelope{T} end

HardEnvelope() = HardEnvelope{Float64}()

# ============================================================================
# Envelope Generation
# ============================================================================

"""
    generate_envelope!(buffer, events, sample_rate, envelope_type)

Fill `buffer` with the amplitude envelope for the given Morse events.
Buffer should be pre-allocated with the correct length.
"""
function generate_envelope!(buffer::Vector{T},
                            events::Vector{TimedMorseEvent{T}},
                            sample_rate::Int,
                            env::RaisedCosineEnvelope{T}) where T<:AbstractFloat
    fill!(buffer, zero(T))
    rise_samples = max(1, round(Int, env.rise_time * sample_rate))

    for event in events
        is_keyed(event.element) || continue

        start_sample = round(Int, event.start_time * sample_rate) + 1
        end_sample = round(Int, (event.start_time + event.duration) * sample_rate)
        end_sample = min(end_sample, length(buffer))
        start_sample = max(1, start_sample)
        start_sample > length(buffer) && continue

        for i in start_sample:end_sample
            # Raised cosine rise
            rise_idx = i - start_sample
            if rise_idx < rise_samples
                envelope_val = T(0.5) * (one(T) - cos(T(π) * rise_idx / rise_samples))
            # Raised cosine fall
            elseif (end_sample - i) < rise_samples
                fall_idx = end_sample - i
                envelope_val = T(0.5) * (one(T) - cos(T(π) * fall_idx / rise_samples))
            else
                envelope_val = one(T)
            end
            buffer[i] = max(buffer[i], envelope_val)
        end
    end

    return buffer
end

function generate_envelope!(buffer::Vector{T},
                            events::Vector{TimedMorseEvent{T}},
                            sample_rate::Int,
                            ::HardEnvelope{T}) where T<:AbstractFloat
    fill!(buffer, zero(T))

    for event in events
        is_keyed(event.element) || continue

        start_sample = round(Int, event.start_time * sample_rate) + 1
        end_sample = round(Int, (event.start_time + event.duration) * sample_rate)
        end_sample = min(end_sample, length(buffer))
        start_sample = max(1, start_sample)
        start_sample > length(buffer) && continue

        for i in start_sample:end_sample
            buffer[i] = one(T)
        end
    end

    return buffer
end

"""
    generate_envelope(events, total_duration, sample_rate; envelope_type) -> Vector{Float64}

Allocate and fill an envelope buffer.
"""
function generate_envelope(events::Vector{TimedMorseEvent{T}},
                           total_dur::T,
                           sample_rate::Int;
                           envelope_type::AbstractEnvelope{T} = RaisedCosineEnvelope{T}(T(0.005))
                           ) where T<:AbstractFloat
    n_samples = round(Int, total_dur * sample_rate)
    buffer = zeros(T, n_samples)
    generate_envelope!(buffer, events, sample_rate, envelope_type)
    return buffer
end
