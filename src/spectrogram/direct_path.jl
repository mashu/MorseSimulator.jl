"""
    MorseSimulator.jl/src/spectrogram/direct_path.jl

Mode 2: Direct (analytic) spectrogram generation.
Bypasses audio waveform synthesis for fast dataset production.
Morse → analytic envelope → linear band power → spectrogram.
"""

using Random

"""
    DirectPath <: AbstractSpectrogramPath

Generate spectrogram analytically without audio synthesis.
Much faster for dataset generation.
"""
struct DirectPath <: AbstractSpectrogramPath end

# ============================================================================
# Main generation
# ============================================================================

"""
    generate_spectrogram(::DirectPath, rng, transcript, scene; kwargs...) -> SpectrogramResult

Generate linear band spectrogram analytically, matching the audio path's physics.

**STFT tone concentration:** A pure CW tone at frequency f concentrates into one
FFT bin with power `(B/2)² × (Σw)²`, while broadband noise has power `σ² × Σ|w|²`
per bin. The ratio `tone_gain = (Σw)² / (4 × Σ|w|²)` (~683 for Hann-4096) gives
the signal its SNR advantage over noise.

**Resolution:** Linear band uses raw FFT bins in [f_min, f_max] → ~(sr/fft_size) Hz
per bin (e.g. ~10.77 Hz at 44.1 kHz, 4096 FFT), so stations 10 Hz apart are separable.
"""
function generate_spectrogram(::DirectPath, rng::AbstractRNG,
                               transcript::Transcript, scene::BandScene;
                               sample_rate::Int = 44100,
                               stft_config::STFTConfig{Float64} = STFTConfig(),
                               linear_band::LinearBand = LinearBand(
                                   fft_size=stft_config.fft_size, sample_rate=sample_rate),
                               envelope_type::AbstractEnvelope{Float64} = RaisedCosineEnvelope())

    scene_events = encode_transcript(rng, transcript, scene)

    total_dur = scene_events.total_duration
    n_samples = round(Int, total_dur * sample_rate)
    n_frames = max(1, (n_samples - stft_config.fft_size) ÷ stft_config.hop_size + 1)

    n_b = n_bins(linear_band)
    spec = zeros(Float64, n_b, n_frames)

    win_energy = sum(abs2, stft_config.window) / stft_config.fft_size
    win_sum  = sum(stft_config.window)
    win_sumsq = sum(abs2, stft_config.window)
    tone_gain = win_sum^2 / (4.0 * win_sumsq)

    channel = ChannelConfig(rng, scene)
    frame_times = [(j - 1) * stft_config.hop_size / sample_rate for j in 1:n_frames]

    for se in scene_events.station_events
        frame_amplitudes = envelope_at_frames(se.events, frame_times, envelope_type)

        prop = scene.propagation
        if rand(rng) < prop.qsb_probability
            f_qsb = prop.qsb_freq_range[1] + rand(rng) * (prop.qsb_freq_range[2] - prop.qsb_freq_range[1])
            depth = clamp(prop.qsb_depth_mean + randn(rng) * 0.1, 0.0, 0.95)
            fading = SinusoidalFading(Float64(depth), f_qsb, Float64(rand(rng) * 2π))
            for j in 1:n_frames
                frame_amplitudes[j] *= fade_factor_at_time(fading, frame_times[j])
            end
        end

        # Tone goes into one FFT bin → one band bin (linear resolution)
        k_fft = fft_bin_for_freq(linear_band, se.tone_freq)
        b = band_bin_for_fft_bin(linear_band, k_fft)
        b == 0 && continue

        for j in 1:n_frames
            power = (se.signal_amplitude * frame_amplitudes[j])^2 * win_energy * tone_gain
            spec[b, j] += power
        end
    end

    for fading in channel.fading_models
        for j in 1:n_frames
            fac = fade_factor_at_time(fading, frame_times[j])
            @inbounds for b in 1:n_b
                spec[b, j] *= fac
            end
        end
    end

    for nm in channel.noise_models
        add_noise_to_band!(rng, spec, nm, win_energy, stft_config.hop_size)
    end

    peak = maximum(spec)
    if peak > 0
        spec ./= peak
    end
    spec = log10.(max.(spec, 1e-10))

    result = SpectrogramResult{Float64}(
        spec, transcript.label, sample_rate,
        total_dur, stft_config, linear_band
    )
    return result, scene_events
end

# ============================================================================
# Envelope at frame times (dispatch on envelope type)
# ============================================================================

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
                frac = (t - (t_start - env.rise_time)) / env.rise_time
                a = 0.5 * (1.0 - cos(π * clamp(frac, 0.0, 1.0)))
            elseif t > t_end
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
# Noise (independent per band bin per frame; linear band = 1 FFT bin per band bin)
# ============================================================================

function add_noise_to_band!(rng::AbstractRNG, spec::AbstractMatrix{Float64},
                           nm::GaussianNoise{T}, win_energy::Float64, hop_size::Int) where T
    base_power = Float64(nm.amplitude^2) * win_energy
    n_b = size(spec, 1)
    n_frames = size(spec, 2)
    for j in 1:n_frames
        @inbounds for b in 1:n_b
            scale = max(0.2, 1.0 + sqrt(2.0) * randn(rng))
            spec[b, j] += base_power * scale
        end
    end
    return nothing
end

function add_noise_to_band!(rng::AbstractRNG, spec::AbstractMatrix{Float64},
                           imp::ImpulsiveNoise{T}, win_energy::Float64, hop_size::Int) where T
    n_b = size(spec, 1)
    n_frames = size(spec, 2)
    frame_imp_prob = 1.0 - (1.0 - Float64(imp.probability))^hop_size
    imp_power = Float64(imp.amplitude^2) * win_energy
    for j in 1:n_frames
        if rand(rng) < frame_imp_prob
            @inbounds for b in 1:n_b
                scale = max(0.2, 1.0 + sqrt(2.0) * randn(rng))
                spec[b, j] += imp_power * scale
            end
        end
    end
    return nothing
end
