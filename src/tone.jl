"""
    CWContestSim/src/signal/tone.jl

CW tone signal generation. Produces a sine wave at the station's
tone frequency, modulated by the Morse envelope.
"""

"""
    generate_tone!(buffer, freq, sample_rate, phase_offset)

Fill `buffer` with a sine wave at frequency `freq` Hz.
Returns the final phase for continuity.
"""
function generate_tone!(buffer::Vector{T}, freq::T,
                        sample_rate::Int,
                        phase_offset::T = zero(T)) where T<:AbstractFloat
    phase_inc = T(2π) * freq / T(sample_rate)
    phase = phase_offset
    @inbounds for i in eachindex(buffer)
        buffer[i] = sin(phase)
        phase += phase_inc
    end
    return phase
end

"""
    generate_cw_signal(envelope, freq, sample_rate; amplitude, freq_offset) -> Vector{T}

Generate the CW signal: amplitude * envelope * sin(2π f t).

# Arguments
- `envelope::Vector{T}` — amplitude envelope
- `freq::T` — CW tone frequency (Hz)
- `sample_rate::Int` — samples per second
- `amplitude::T` — signal amplitude
- `freq_offset::T` — small frequency offset for realism
"""
function generate_cw_signal(envelope::Vector{T}, freq::T,
                            sample_rate::Int;
                            amplitude::T = one(T),
                            freq_offset::T = zero(T)) where T<:AbstractFloat
    n = length(envelope)
    signal = Vector{T}(undef, n)
    actual_freq = freq + freq_offset
    phase_inc = T(2π) * actual_freq / T(sample_rate)
    phase = zero(T)

    @inbounds for i in 1:n
        signal[i] = amplitude * envelope[i] * sin(phase)
        phase += phase_inc
    end

    return signal
end

"""
    generate_station_signal(station_events, total_duration, sample_rate; envelope_type) -> Vector{Float64}

Generate the complete CW signal for one station.
"""
function generate_station_signal(se::StationMorseEvents{T},
                                 total_dur::T,
                                 sample_rate::Int;
                                 envelope_type::AbstractEnvelope{T} = RaisedCosineEnvelope{T}(T(0.005))
                                 ) where T<:AbstractFloat
    env = generate_envelope(se.events, total_dur, sample_rate; envelope_type=envelope_type)
    return generate_cw_signal(env, T(se.tone_freq), sample_rate;
                              amplitude=T(se.signal_amplitude))
end
