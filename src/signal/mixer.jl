"""
    MorseSimulator.jl/src/signal/mixer.jl

Multi-station signal mixer. Generates individual station signals
and sums them linearly, then applies channel effects.
"""

using Random

"""
    MixedSignal{T<:AbstractFloat}

Result of mixing multiple station signals.

# Fields
- `samples::Vector{T}` — the mixed audio samples
- `sample_rate::Int` — sample rate in Hz
- `duration::T` — total duration in seconds
- `label::String` — training label text
"""
struct MixedSignal{T<:AbstractFloat}
    samples::Vector{T}
    sample_rate::Int
    duration::T
    label::String
end

"""
    mix_signals(rng, scene_events, scene; sample_rate, envelope_type) -> MixedSignal

Generate and mix CW signals for all stations, apply channel effects.
"""
function mix_signals(rng::AbstractRNG,
                     scene_events::SceneMorseEvents{T},
                     scene::BandScene;
                     sample_rate::Int = 44100,
                     envelope_type::AbstractEnvelope{T} = RaisedCosineEnvelope{T}(T(0.005))
                     ) where T<:AbstractFloat
    n_samples = round(Int, scene_events.total_duration * sample_rate)
    mixed = zeros(T, n_samples)

    # Generate per-station fading
    channel = ChannelConfig(rng, scene)

    for se in scene_events.station_events
        station_sig = generate_station_signal(se, scene_events.total_duration,
                                              sample_rate; envelope_type=envelope_type)

        # Apply per-station fading if configured
        prop = scene.propagation
        if rand(rng) < prop.qsb_probability
            f_qsb = prop.qsb_freq_range[1] + rand(rng) * (prop.qsb_freq_range[2] - prop.qsb_freq_range[1])
            depth = clamp(prop.qsb_depth_mean + randn(rng) * 0.1, 0.0, 0.95)
            fading = SinusoidalFading{T}(T(depth), T(f_qsb), T(rand(rng) * 2π))
            apply_fading!(station_sig, fading, sample_rate)
        end

        # Sum into mix
        n = min(length(mixed), length(station_sig))
        @inbounds for i in 1:n
            mixed[i] += station_sig[i]
        end
    end

    # Apply channel noise
    apply_channel!(rng, mixed, channel, sample_rate)

    # Normalize to prevent clipping
    peak = maximum(abs, mixed)
    if peak > zero(T)
        scale = T(0.95) / peak
        mixed .*= scale
    end

    return MixedSignal{T}(mixed, sample_rate, scene_events.total_duration,
                           scene_events.label)
end

mix_signals(scene_events::SceneMorseEvents, scene::BandScene; kwargs...) =
    mix_signals(Random.default_rng(), scene_events, scene; kwargs...)
