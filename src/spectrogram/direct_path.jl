"""
    MorseSimulator.jl/src/spectrogram/direct_path.jl

Mode 2: Direct (analytic) mel-spectrogram generation.
Bypasses audio waveform synthesis for fast dataset production.
Morse → analytic envelope → mel-frequency energy → spectrogram.
"""

using Random

"""
    DirectPath <: AbstractSpectrogramPath

Generate mel-spectrogram analytically without audio synthesis.
Much faster for dataset generation.
"""
struct DirectPath <: AbstractSpectrogramPath end

"""
    generate_spectrogram(::DirectPath, rng, transcript, scene; kwargs...) -> SpectrogramResult

Generate mel-spectrogram analytically.
Models the STFT of a windowed sinusoid modulated by the Morse envelope.

For a CW tone at frequency f with envelope A(t):
  Power at mel bin m ≈ A²(t_frame) × H_m(f)²

where H_m(f) is the mel filter weight at the tone frequency.
"""
function generate_spectrogram(::DirectPath, rng::AbstractRNG,
                               transcript::Transcript, scene::BandScene;
                               sample_rate::Int = 44100,
                               stft_config::STFTConfig{Float64} = STFTConfig(),
                               filterbank::MelFilterbank{Float64} = MelFilterbank(
                                   fft_size=stft_config.fft_size, sample_rate=sample_rate),
                               envelope_type::AbstractEnvelope{Float64} = RaisedCosineEnvelope())

    # Encode transcript to Morse events
    scene_events = encode_transcript(rng, transcript, scene)

    # Frame timing
    total_dur = scene_events.total_duration
    n_samples = round(Int, total_dur * sample_rate)
    n_frames = max(1, (n_samples - stft_config.fft_size) ÷ stft_config.hop_size + 1)

    # Initialize mel spectrogram
    mel_spec = zeros(Float64, filterbank.n_filters, n_frames)

    # Precompute window energy for normalization
    win_energy = sum(abs2, stft_config.window) / stft_config.fft_size

    channel = ChannelConfig(rng, scene)
    frame_times = [(j - 1) * stft_config.hop_size / sample_rate for j in 1:n_frames]

    # For each station, compute analytic contribution
    for se in scene_events.station_events
        frame_amplitudes = envelope_at_frames(se.events, frame_times, envelope_type)

        # Per-station QSB fading (same as mixer)
        prop = scene.propagation
        if rand(rng) < prop.qsb_probability
            f_qsb = prop.qsb_freq_range[1] + rand(rng) * (prop.qsb_freq_range[2] - prop.qsb_freq_range[1])
            depth = clamp(prop.qsb_depth_mean + randn(rng) * 0.1, 0.0, 0.95)
            fading = SinusoidalFading(Float64(depth), f_qsb, Float64(rand(rng) * 2π))
            for j in 1:n_frames
                frame_amplitudes[j] *= fade_factor_at_time(fading, frame_times[j])
            end
        end

        # Compute mel filter response at station's tone frequency
        tone_bin = round(Int, se.tone_freq / (sample_rate / stft_config.fft_size)) + 1
        tone_bin = clamp(tone_bin, 1, stft_config.fft_size ÷ 2 + 1)
        mel_response = filterbank.filters[:, tone_bin]

        # Fill spectrogram: power ∝ amplitude² × filter_weight² × window_energy
        for j in 1:n_frames
            power = (se.signal_amplitude * frame_amplitudes[j])^2 * win_energy
            @inbounds for m in 1:filterbank.n_filters
                mel_spec[m, j] += power * mel_response[m]^2
            end
        end
    end

    # Global fading then noise (same order as apply_channel!)
    for fading in channel.fading_models
        for j in 1:n_frames
            fac = fade_factor_at_time(fading, frame_times[j])
            @inbounds for m in 1:filterbank.n_filters
                mel_spec[m, j] *= fac
            end
        end
    end

    # Noise: constant power from all models (Gaussian contributes; Impulsive 0)
    constant_power = sum(nm -> constant_noise_power(nm), channel.noise_models) * win_energy
    for j in 1:n_frames
        @inbounds for m in 1:filterbank.n_filters
            mel_spec[m, j] += constant_power
        end
    end
    for nm in channel.noise_models
        add_noise_to_mel!(rng, mel_spec, nm, win_energy, stft_config.hop_size)
    end

    # Normalize peak (approximate match to audio path)
    peak = maximum(mel_spec)
    if peak > 0
        mel_spec ./= peak
    end

    # Log compression
    mel_spec = log10.(max.(mel_spec, 1e-10))

    return SpectrogramResult{Float64}(
        mel_spec, transcript.label, sample_rate,
        total_dur, stft_config, filterbank
    )
end

# ============================================================================
# Envelope at frame times (dispatch on envelope type)
# ============================================================================

"""
    envelope_at_frames(events, frame_times, envelope_type) -> Vector{Float64}

Compute envelope amplitude at each frame center time.
"""
function envelope_at_frames(events::Vector{TimedMorseEvent{Float64}},
                              frame_times::Vector{Float64},
                              env::RaisedCosineEnvelope{Float64})
    n_frames = length(frame_times)
    amplitudes = zeros(Float64, n_frames)

    for event in events
        is_keyed(event.element) || continue
        t_start = event.start_time
        t_end = t_start + event.duration

        for j in 1:n_frames
            t = frame_times[j]
            (t < t_start - env.rise_time || t > t_end + env.rise_time) && continue

            if t < t_start
                # Rising edge
                frac = (t - (t_start - env.rise_time)) / env.rise_time
                a = 0.5 * (1.0 - cos(π * clamp(frac, 0.0, 1.0)))
            elseif t > t_end
                # Falling edge
                frac = (t - t_end) / env.rise_time
                a = 0.5 * (1.0 + cos(π * clamp(frac, 0.0, 1.0)))
            else
                a = 1.0
            end
            amplitudes[j] = max(amplitudes[j], a)
        end
    end

    return amplitudes
end

function envelope_at_frames(events::Vector{TimedMorseEvent{Float64}},
                             frame_times::Vector{Float64},
                             ::HardEnvelope{Float64})
    n_frames = length(frame_times)
    amplitudes = zeros(Float64, n_frames)

    for event in events
        is_keyed(event.element) || continue
        t_start = event.start_time
        t_end = t_start + event.duration

        for j in 1:n_frames
            t = frame_times[j]
            if t >= t_start && t <= t_end
                amplitudes[j] = max(amplitudes[j], 1.0)
            end
        end
    end

    return amplitudes
end

# ============================================================================
# Noise applied to mel (dispatch: Gaussian already in constant; Impulsive adds bursts)
# ============================================================================

"""Gaussian: constant already added to all frames; no per-frame add."""
add_noise_to_mel!(::AbstractRNG, ::AbstractMatrix, ::GaussianNoise, _, _) = nothing

"""Impulsive: add random burst power per frame (P(impulse in frame) from per-sample prob)."""
function add_noise_to_mel!(rng::AbstractRNG, mel_spec::AbstractMatrix{Float64},
                           imp::ImpulsiveNoise{T}, win_energy::Float64, hop_size::Int) where T
    n_frames = size(mel_spec, 2)
    n_filters = size(mel_spec, 1)
    frame_imp_prob = 1.0 - (1.0 - Float64(imp.probability))^hop_size
    imp_power = Float64(imp.amplitude^2) * win_energy
    for j in 1:n_frames
        if rand(rng) < frame_imp_prob
            @inbounds for m in 1:n_filters
                mel_spec[m, j] += imp_power
            end
        end
    end
    return nothing
end
