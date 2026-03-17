"""
    MorseSimulator.jl/src/spectrogram/audio_path.jl

Mode 1: Audio path for spectrogram generation.
Morse → waveform → STFT → linear band power spectrogram.
This path produces audio that can be saved for inspection.
"""

"""
    AudioPath <: AbstractSpectrogramPath

Generate spectrogram via full audio waveform synthesis.
"""
struct AudioPath <: AbstractSpectrogramPath end

"""
    SpectrogramResult{T<:AbstractFloat}

Result of spectrogram generation (linear band, not mel).

# Fields
- `spectrogram::Matrix{T}` — (n_bins × n_frames), log10 scale
- `label::String` — training label
- `sample_rate::Int` — audio sample rate
- `duration::T` — total duration in seconds
- `stft_config::STFTConfig{T}` — STFT parameters used
- `linear_band::LinearBand` — band configuration
"""
struct SpectrogramResult{T<:AbstractFloat}
    spectrogram::Matrix{T}
    label::String
    sample_rate::Int
    duration::T
    stft_config::STFTConfig{T}
    linear_band::LinearBand
end

"""
    generate_spectrogram(::AudioPath, rng, transcript, scene; kwargs...) -> (SpectrogramResult, MixedSignal)

Generate linear band spectrogram via audio synthesis.
Also returns the mixed signal for audio inspection.
"""
function generate_spectrogram(::AudioPath, rng::AbstractRNG,
                               transcript::Transcript, scene::BandScene;
                               sample_rate::Int = 44100,
                               stft_config::STFTConfig{Float64} = STFTConfig(),
                               linear_band::LinearBand = LinearBand(
                                   fft_size=stft_config.fft_size, sample_rate=sample_rate),
                               envelope_type::AbstractEnvelope{Float64} = RaisedCosineEnvelope())

    # Encode transcript to Morse events (same as direct path)
    scene_events = encode_transcript(rng, transcript, scene)

    # Generate mixed audio signal
    mixed = mix_signals(rng, scene_events, scene;
                        sample_rate=sample_rate, envelope_type=envelope_type)

    # Compute STFT
    stft_result = compute_stft(mixed.samples, stft_config)

    # Power spectrogram
    pspec = power_spectrogram(stft_result)

    # Slice to band (linear, no mel)
    band_spec = pspec[linear_band.bin_lo:linear_band.bin_hi, :]

    # Normalize peak (match direct path so both outputs on same scale)
    peak = maximum(band_spec)
    if peak > 0
        band_spec = band_spec ./ peak
    end

    # Log compression
    band_spec = log10.(max.(band_spec, 1e-10))

    result = SpectrogramResult{Float64}(
        band_spec, transcript.label, sample_rate,
        mixed.duration, stft_config, linear_band
    )

    return result, mixed, scene_events
end
