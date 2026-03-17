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

function apply_noise!(rng::AbstractRNG, signal::Vector{T},
                      noise::GaussianNoise{T}) where T<:AbstractFloat
    @inbounds for i in eachindex(signal)
        signal[i] += noise.amplitude * T(randn(rng))
    end
    return signal
end

function apply_noise!(rng::AbstractRNG, signal::Vector{T},
                      noise::ImpulsiveNoise{T}) where T<:AbstractFloat
    @inbounds for i in eachindex(signal)
        if rand(rng) < noise.probability
            signal[i] += noise.amplitude * (T(2) * rand(rng, T) - one(T))
        end
    end
    return signal
end

"""
    constant_noise_power(nm) -> Float64

Noise power (amplitude²) that is constant in time. Gaussian: amp²; Impulsive: 0
(impulses are applied per-frame in mel path).
"""
constant_noise_power(nm::GaussianNoise{T}) where T = Float64(nm.amplitude^2)
constant_noise_power(nm::ImpulsiveNoise{T}) where T = zero(Float64)

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
    fade_factor_at_time(fading, t) -> Float64

Fade multiplier at time t (seconds). Single source of truth for both
waveform (apply_fading!) and direct mel path.
"""
function fade_factor_at_time(fading::SinusoidalFading{T}, t::Real) where T
    phase = fading.phase + T(2π) * fading.frequency * T(t)
    Float64(one(T) - fading.depth * T(0.5) * (one(T) + sin(phase)))
end

function fade_factor_at_time(fading::RayleighFading{T}, t::Real) where T
    n_paths = 8
    freqs = range(-fading.bandwidth, fading.bandwidth, length=n_paths)
    phases = range(zero(T), T(2π), length=n_paths + 1)[1:end-1]
    re = zero(Float64)
    im = zero(Float64)
    t_ = Float64(t)
    for k in 1:n_paths
        angle = 2π * freqs[k] * t_ + Float64(phases[k])
        re += cos(angle)
        im += sin(angle)
    end
    mag = sqrt(re^2 + im^2) / n_paths
    Float64(max(0.0, one(T) - fading.depth * (one(T) - T(mag))))
end

function apply_fading!(signal::Vector{T}, fading::SinusoidalFading{T},
                       sample_rate::Int) where T<:AbstractFloat
    @inbounds for i in eachindex(signal)
        t = (i - 1) / sample_rate
        signal[i] *= T(fade_factor_at_time(fading, t))
    end
    return signal
end

function apply_fading!(signal::Vector{T}, fading::RayleighFading{T},
                       sample_rate::Int) where T<:AbstractFloat
    @inbounds for i in eachindex(signal)
        t = (i - 1) / sample_rate
        signal[i] *= T(fade_factor_at_time(fading, t))
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

Create a channel configuration from a band scene.
Noise amplitude is derived from the scene's SNR target via `noise_amplitude(scene)`.
"""
function ChannelConfig(rng::AbstractRNG, scene::BandScene)
    prop = scene.propagation
    Tf = Float64

    noise_amp = Tf(noise_amplitude(scene))

    noise_models = AbstractNoiseModel{Tf}[GaussianNoise{Tf}(noise_amp)]
    if rand(rng) < 0.3
        push!(noise_models, ImpulsiveNoise{Tf}(Tf(0.0001), noise_amp * Tf(5.0)))
    end

    fading_models = AbstractFadingModel{Tf}[]
    if rand(rng) < prop.qsb_probability
        f_qsb = prop.qsb_freq_range[1] + rand(rng) * (prop.qsb_freq_range[2] - prop.qsb_freq_range[1])
        depth = clamp(prop.qsb_depth_mean + randn(rng) * 0.1, 0.0, 0.95)
        phase = rand(rng) * Tf(2π)
        push!(fading_models, SinusoidalFading{Tf}(Tf(depth), Tf(f_qsb), phase))
    end

    ChannelConfig{Tf}(noise_models, fading_models)
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
