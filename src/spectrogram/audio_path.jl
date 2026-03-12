"""
    MorseSimulator.jl/src/spectrogram/audio_path.jl

Mode 1: Audio path for mel-spectrogram generation.
Morse → waveform → STFT → mel spectrogram.
This path produces audio that can be saved for inspection.
"""

"""
    AudioPath <: AbstractSpectrogramPath

Generate mel-spectrogram via full audio waveform synthesis.
"""
struct AudioPath <: AbstractSpectrogramPath end

"""
    SpectrogramResult{T<:AbstractFloat}

Result of mel-spectrogram generation.

# Fields
- `mel_spectrogram::Matrix{T}` — mel spectrogram (n_mels × n_frames)
- `label::String` — training label
- `sample_rate::Int` — audio sample rate
- `duration::T` — total duration in seconds
- `stft_config::STFTConfig{T}` — STFT parameters used
- `filterbank::MelFilterbank{T}` — mel filterbank used
"""
struct SpectrogramResult{T<:AbstractFloat}
    mel_spectrogram::Matrix{T}
    label::String
    sample_rate::Int
    duration::T
    stft_config::STFTConfig{T}
    filterbank::MelFilterbank{T}
end

"""
    generate_spectrogram(::AudioPath, rng, transcript, scene; kwargs...) -> (SpectrogramResult, MixedSignal)

Generate mel-spectrogram via audio synthesis.
Also returns the mixed signal for audio inspection.
"""
function generate_spectrogram(::AudioPath, rng::AbstractRNG,
                               transcript::Transcript, scene::BandScene;
                               sample_rate::Int = 44100,
                               stft_config::STFTConfig{Float64} = STFTConfig(),
                               filterbank::MelFilterbank{Float64} = MelFilterbank(
                                   fft_size=stft_config.fft_size, sample_rate=sample_rate),
                               envelope_type::AbstractEnvelope{Float64} = RaisedCosineEnvelope())

    # Encode transcript to Morse events
    scene_events = encode_transcript(rng, transcript, scene)

    # Generate mixed audio signal
    mixed = mix_signals(rng, scene_events, scene;
                        sample_rate=sample_rate, envelope_type=envelope_type)

    # Compute STFT
    stft_result = compute_stft(mixed.samples, stft_config)

    # Power spectrogram
    pspec = power_spectrogram(stft_result)

    # Apply mel filterbank
    mel_spec = apply_filterbank(filterbank, pspec)

    # Normalize peak (match direct path so both outputs on same scale)
    peak = maximum(mel_spec)
    if peak > 0
        mel_spec ./= peak
    end

    # Log compression
    mel_spec = log10.(max.(mel_spec, 1e-10))

    result = SpectrogramResult{Float64}(
        mel_spec, transcript.label, sample_rate,
        mixed.duration, stft_config, filterbank
    )

    return result, mixed
end
