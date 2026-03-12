"""
    MorseSimulator.jl/src/signal/channel.jl

Channel model applying realistic propagation effects to CW signals.
Effects include additive noise, QSB fading, impulsive noise,
and frequency drift.
"""

using Random, Distributions

# ============================================================================
# Noise Models
# ============================================================================

"""
    GaussianNoise{T} <: AbstractNoiseModel{T}

Additive white Gaussian noise.

# Fields
- `amplitude::T` — noise RMS amplitude
"""
struct GaussianNoise{T<:AbstractFloat} <: AbstractNoiseModel{T}
    amplitude::T
end

"""
    ImpulsiveNoise{T} <: AbstractNoiseModel{T}

Occasional impulsive noise (QRN).

# Fields
- `probability::T` — probability of impulse per sample
- `amplitude::T` — impulse amplitude
"""
struct ImpulsiveNoise{T<:AbstractFloat} <: AbstractNoiseModel{T}
    probability::T
    amplitude::T
end

"""
    apply_noise!(rng, signal, noise::GaussianNoise)

Add Gaussian noise to signal in-place.
"""
function apply_noise!(rng::AbstractRNG, signal::Vector{T},
                      noise::GaussianNoise{T}) where T<:AbstractFloat
    @inbounds for i in eachindex(signal)
        signal[i] += noise.amplitude * T(randn(rng))
    end
    return signal
end

"""
    apply_noise!(rng, signal, noise::ImpulsiveNoise)

Add impulsive noise to signal in-place.
"""
function apply_noise!(rng::AbstractRNG, signal::Vector{T},
                      noise::ImpulsiveNoise{T}) where T<:AbstractFloat
    @inbounds for i in eachindex(signal)
        if rand(rng) < noise.probability
            signal[i] += noise.amplitude * (T(2) * rand(rng, T) - one(T))
        end
    end
    return signal
end

# ============================================================================
# Fading Models (QSB)
# ============================================================================

"""
    SinusoidalFading{T} <: AbstractFadingModel{T}

Sinusoidal QSB fading.

# Fields
- `depth::T` — fading depth (0 = no fading, 1 = full fade)
- `frequency::T` — fading rate in Hz
- `phase::T` — initial phase
"""
struct SinusoidalFading{T<:AbstractFloat} <: AbstractFadingModel{T}
    depth::T
    frequency::T
    phase::T
end

"""
    RayleighFading{T} <: AbstractFadingModel{T}

Rayleigh fading model for more realistic deep fades.

# Fields
- `bandwidth::T` — Doppler bandwidth (Hz)
- `depth::T` — fading depth
"""
struct RayleighFading{T<:AbstractFloat} <: AbstractFadingModel{T}
    bandwidth::T
    depth::T
end

"""
    apply_fading!(signal, fading::SinusoidalFading, sample_rate)

Apply sinusoidal fading to signal in-place.
"""
function apply_fading!(signal::Vector{T}, fading::SinusoidalFading{T},
                       sample_rate::Int) where T<:AbstractFloat
    phase_inc = T(2π) * fading.frequency / T(sample_rate)
    phase = fading.phase
    @inbounds for i in eachindex(signal)
        fade_factor = one(T) - fading.depth * T(0.5) * (one(T) + sin(phase))
        signal[i] *= fade_factor
        phase += phase_inc
    end
    return signal
end

"""
    apply_fading!(signal, fading::RayleighFading, sample_rate)

Apply Rayleigh-like fading using sum of sinusoids approximation.
"""
function apply_fading!(signal::Vector{T}, fading::RayleighFading{T},
                       sample_rate::Int) where T<:AbstractFloat
    n_paths = 8
    freqs = range(-fading.bandwidth, fading.bandwidth, length=n_paths)
    phases = range(zero(T), T(2π), length=n_paths+1)[1:end-1]

    @inbounds for i in eachindex(signal)
        t = T(i - 1) / T(sample_rate)
        re = zero(T)
        im = zero(T)
        for k in 1:n_paths
            angle = T(2π) * freqs[k] * t + phases[k]
            re += cos(angle)
            im += sin(angle)
        end
        mag = sqrt(re^2 + im^2) / T(n_paths)
        fade_factor = one(T) - fading.depth * (one(T) - mag)
        signal[i] *= max(zero(T), fade_factor)
    end
    return signal
end

# ============================================================================
# Channel Configuration
# ============================================================================

"""
    ChannelConfig{T<:AbstractFloat}

Complete channel configuration.

# Fields
- `noise_models::Vector{AbstractNoiseModel{T}}`
- `fading_models::Vector{AbstractFadingModel{T}}`
"""
struct ChannelConfig{T<:AbstractFloat}
    noise_models::Vector{<:AbstractNoiseModel{T}}
    fading_models::Vector{<:AbstractFadingModel{T}}
end

"""
    ChannelConfig(rng, scene) -> ChannelConfig{Float64}

Create a channel configuration from a band scene's propagation conditions.
"""
function ChannelConfig(rng::AbstractRNG, scene::BandScene)
    prop = scene.propagation
    T = Float64

    noise_models = AbstractNoiseModel{T}[]
    # Gaussian noise based on noise floor
    noise_amp = T(10.0^(scene.noise_floor_db / 20.0))
    push!(noise_models, GaussianNoise{T}(noise_amp))

    # Optional impulsive noise
    if rand(rng) < 0.3
        push!(noise_models, ImpulsiveNoise{T}(T(0.0001), noise_amp * T(5.0)))
    end

    fading_models = AbstractFadingModel{T}[]
    if rand(rng) < prop.qsb_probability
        f_qsb = prop.qsb_freq_range[1] + rand(rng) * (prop.qsb_freq_range[2] - prop.qsb_freq_range[1])
        depth = prop.qsb_depth_mean + randn(rng) * 0.1
        depth = clamp(depth, 0.0, 0.95)
        phase = rand(rng) * T(2π)
        push!(fading_models, SinusoidalFading{T}(T(depth), T(f_qsb), phase))
    end

    return ChannelConfig{T}(noise_models, fading_models)
end

"""
    apply_channel!(rng, signal, channel, sample_rate)

Apply all channel effects to a signal in-place.
"""
function apply_channel!(rng::AbstractRNG, signal::Vector{T},
                        channel::ChannelConfig{T},
                        sample_rate::Int) where T<:AbstractFloat
    for fading in channel.fading_models
        apply_fading!(signal, fading, sample_rate)
    end
    for noise in channel.noise_models
        apply_noise!(rng, signal, noise)
    end
    return signal
end
