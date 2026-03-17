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

# ============================================================================
# Precomputation helpers
# ============================================================================

"""
    mel_noise_bandwidth(filterbank) -> Vector{Float64}

Per-mel-bin noise bandwidth weight: `bw[m] = Σ_k filters[m,k]`.

In a real STFT, white noise with power σ² per FFT bin produces mel noise
`mel[m] = Σ_k filters[m,k] * |noise[k]|²`. The expected value is
`σ² × Σ|w|² × Σ_k filters[m,k]`, so wider filters accumulate more noise.
"""
function mel_noise_bandwidth(filterbank::MelFilterbank{Float64})
    n_filters = filterbank.n_filters
    n_bins = size(filterbank.filters, 2)
    bw = zeros(Float64, n_filters)
    @inbounds for m in 1:n_filters
        for k in 1:n_bins
            bw[m] += filterbank.filters[m, k]
        end
    end
    return bw
end

# ============================================================================
# Main generation
# ============================================================================

"""
    generate_spectrogram(::DirectPath, rng, transcript, scene; kwargs...) -> SpectrogramResult

Generate mel-spectrogram analytically, matching the audio path's physics.

**STFT tone concentration:** A pure CW tone at frequency f concentrates into one
FFT bin with power `(B/2)² × (Σw)²`, while broadband noise has power `σ² × Σ|w|²`
per bin. The ratio `tone_gain = (Σw)² / (4 × Σ|w|²)` (~683 for Hann-4096) gives
the signal its SNR advantage over noise, making Morse dots/dashes clearly visible
above the noise floor. Without this factor, signal and noise have the same power
and the Morse signal is invisible.

**Noise model:** Each mel bin gets an independent noise draw per frame, matching
the audio path where each FFT bin has independent noise that passes through the
mel filterbank.
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

    # Precompute window energy (power normalization)
    win_energy = sum(abs2, stft_config.window) / stft_config.fft_size

    # STFT tone concentration factor.
    # In a real STFT, a pure tone at bin k₀ has power |X[k₀]|² = (B/2)² × (Σw)²
    # while noise per FFT bin has E[|X|²] = σ² × Σ|w|².
    # The tone_gain = (Σw)² / (4 × Σ|w|²) corrects the signal power so the
    # signal-to-noise ratio matches the audio path after peak normalization + log.
    win_sum  = sum(stft_config.window)
    win_sumsq = sum(abs2, stft_config.window)
    tone_gain = win_sum^2 / (4.0 * win_sumsq)

    channel = ChannelConfig(rng, scene)
    frame_times = [(j - 1) * stft_config.hop_size / sample_rate for j in 1:n_frames]

    # Precompute per-bin noise bandwidth weights
    noise_bw = mel_noise_bandwidth(filterbank)

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

        # Fill spectrogram with tone-concentrated signal power.
        # Mel filterbank applies LINEARLY to the power spectrum:
        #   mel[m] = filters[m, k₀] × |X[k₀]|²
        # The tone_gain accounts for STFT concentration of the tone into one bin.
        for j in 1:n_frames
            power = (se.signal_amplitude * frame_amplitudes[j])^2 * win_energy * tone_gain
            @inbounds for m in 1:filterbank.n_filters
                mel_spec[m, j] += power * mel_response[m]
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

    # Noise: independent per mel bin per frame, matching real STFT behavior
    # where each FFT bin has independent noise through the mel filterbank.
    for nm in channel.noise_models
        add_noise_to_mel!(rng, mel_spec, nm, win_energy, stft_config.hop_size; noise_bw)
    end

    # Normalize peak (approximate match to audio path)
    peak = maximum(mel_spec)
    if peak > 0
        mel_spec ./= peak
    end

    # Log compression
    mel_spec = log10.(max.(mel_spec, 1e-10))

    result = SpectrogramResult{Float64}(
        mel_spec, transcript.label, sample_rate,
        total_dur, stft_config, filterbank
    )
    return result, scene_events
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
# Noise applied to mel (independent per bin per frame, matching real STFT)
# ============================================================================

"""
Gaussian: independent noise per mel bin per frame.

In a real STFT, each FFT bin gets independent noise. After the mel filterbank
(which covers ~1–2 FFT bins per filter at fft_size=4096), each mel bin has
its own independent noise realization. This per-bin independence is what creates
the noise texture visible in real spectrograms and lets the encoder distinguish
signal bins from noise bins within a single frame.

`noise_bw[m]` scales by filter bandwidth so wider filters accumulate more noise.
"""
function add_noise_to_mel!(rng::AbstractRNG, mel_spec::AbstractMatrix{Float64},
                           nm::GaussianNoise{T}, win_energy::Float64, hop_size::Int;
                           noise_bw::Vector{Float64} = ones(Float64, size(mel_spec, 1))) where T
    base_power = Float64(nm.amplitude^2) * win_energy
    n_frames = size(mel_spec, 2)
    n_filters = size(mel_spec, 1)
    for j in 1:n_frames
        @inbounds for m in 1:n_filters
            # Independent exponential-like draw per mel bin per frame
            # (matches chi-squared distribution of |FFT_noise|² per bin)
            scale = max(0.2, 1.0 + sqrt(2.0) * randn(rng))
            mel_spec[m, j] += base_power * scale * noise_bw[m]
        end
    end
    return nothing
end

"""
Impulsive: burst power in random frames with per-bin variation.
"""
function add_noise_to_mel!(rng::AbstractRNG, mel_spec::AbstractMatrix{Float64},
                           imp::ImpulsiveNoise{T}, win_energy::Float64, hop_size::Int;
                           noise_bw::Vector{Float64} = ones(Float64, size(mel_spec, 1))) where T
    n_frames = size(mel_spec, 2)
    n_filters = size(mel_spec, 1)
    frame_imp_prob = 1.0 - (1.0 - Float64(imp.probability))^hop_size
    imp_power = Float64(imp.amplitude^2) * win_energy
    for j in 1:n_frames
        if rand(rng) < frame_imp_prob
            @inbounds for m in 1:n_filters
                scale = max(0.2, 1.0 + sqrt(2.0) * randn(rng))
                mel_spec[m, j] += imp_power * scale * noise_bw[m]
            end
        end
    end
    return nothing
end

