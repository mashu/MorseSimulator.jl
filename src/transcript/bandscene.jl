"""
    MorseSimulator.jl/src/transcript/bandscene.jl

Band scene model representing the radio environment in which
stations operate. Controls propagation, noise, fading, and
station density parameters.

Noise level is derived from a sampled waveform SNR so the signal-to-noise
ratio is always in a learnable range. The STFT tone concentration (~28 dB
for Hann-4096) lifts the spectrogram SNR well above the waveform SNR.
"""

using Random, Distributions

"""
    PropagationCondition

Propagation conditions affecting signal strength and fading.
"""
struct PropagationCondition
    path_loss_mean::Float64      # Mean path loss in dB
    path_loss_sigma::Float64     # Variability of path loss
    qsb_probability::Float64    # Probability of QSB fading
    qsb_depth_mean::Float64     # Mean fading depth (0-1)
    qsb_freq_range::Tuple{Float64,Float64}  # QSB frequency range (Hz)
end

good_propagation() = PropagationCondition(20.0, 5.0, 0.2, 0.1, (0.05, 0.3))
moderate_propagation() = PropagationCondition(40.0, 10.0, 0.5, 0.3, (0.1, 0.5))
poor_propagation() = PropagationCondition(60.0, 15.0, 0.8, 0.6, (0.2, 1.0))

"""
    BandScene{C<:AbstractContest, M<:AbstractConversationMode}

The simulated radio band environment.

# Fields
- `contest::C` — active contest type
- `mode::M` — conversation mode
- `propagation::PropagationCondition` — propagation conditions
- `snr_db::Float64` — waveform SNR in dB (spectrogram SNR ≈ snr_db + 28 dB)
- `station_density::Float64` — expected number of active stations
- `collision_probability::Float64` — probability of signal overlap
- `stations::Vector{Station}` — stations present on the band
"""
struct BandScene{C<:AbstractContest, M<:AbstractConversationMode}
    contest::C
    mode::M
    propagation::PropagationCondition
    snr_db::Float64
    station_density::Float64
    collision_probability::Float64
    stations::Vector{<:Station}
end

"""
    BandScene(rng; num_stations, contest, mode, propagation, snr_range, min_tone_separation_Hz) -> BandScene

Construct a BandScene with SNR-controlled noise.

Stations are assigned tones exactly `min_tone_separation_Hz` apart (ref_freq, ref_freq ± step,
ref_freq ± 2*step, …) so they land in distinct spectrogram bins. With linear band, bin width
is sample_rate/fft_size (e.g. ~10.77 Hz at 44.1 kHz, 4096 FFT). Tones are assigned by
round(f * fft_size / sr), so they end up in the same bin only if closer than ~½ bin width.
Use min_tone_separation_Hz >= sample_rate/fft_size (e.g. 11 Hz for 44.1k/4096) to guarantee
separate bins; 10 Hz is typical and sufficient for that config.

A target waveform SNR is sampled uniformly from `snr_range` (default: (-20, 10) dB). Noise
amplitude is derived on demand via `noise_amplitude(scene)` from station amplitudes and propagation.

The STFT tone concentration adds ~28 dB on top (Hann-4096), giving:

| Waveform SNR | Spectrogram SNR | Difficulty |
|:-------------|:----------------|:-----------|
| -20 dB       | ~8 dB           | Hard       |
| -10 dB       | ~18 dB          | Medium     |
|   0 dB       | ~28 dB          | Easy       |
| +10 dB       | ~38 dB          | Very easy  |
"""
function BandScene(rng::AbstractRNG;
                   num_stations::Int = rand(rng, 2:6),
                   contest::AbstractContest = random_contest(rng),
                   mode::AbstractConversationMode = random_mode(rng),
                   propagation::PropagationCondition = moderate_propagation(),
                   snr_range::Tuple{Float64,Float64} = (-20.0, 10.0),
                   min_tone_separation_Hz::Float64 = 10.0)

    callsigns = generate_callsigns(rng, num_stations)
    styles = select_styles(rng, mode, num_stations)
    ref_freq = clamp(600.0 + 80.0 * randn(rng), 400.0, 700.0)

    # Assign tones with minimum separation so stations show in distinct spectrogram bins.
    step = min_tone_separation_Hz
    caller_tone = ref_freq
    n_other = num_stations - 1
    other_tones = Float64[]
    for k in 1:max(n_other, 4)
        push!(other_tones, clamp(ref_freq + k * step, TONE_BAND[1], TONE_BAND[2]))
        push!(other_tones, clamp(ref_freq - k * step, TONE_BAND[1], TONE_BAND[2]))
    end
    # Exclude caller tone so no other station shares it
    other_tones = unique!(sort!([t for t in other_tones if t != ref_freq]))
    while length(other_tones) < n_other
        k = length(other_tones) + 1
        push!(other_tones, clamp(ref_freq + k * step, TONE_BAND[1], TONE_BAND[2]))
        other_tones = unique!(sort!(other_tones))
    end
    other_tones = shuffle(rng, other_tones[1:n_other])

    tone_list = [caller_tone; other_tones]

    stations = [
        Station(rng, call, style; tone_freq=tone_list[i], is_caller=(i == 1))
        for (i, (call, style)) in enumerate(zip(callsigns, styles))
    ]

    snr_db = snr_range[1] + rand(rng) * (snr_range[2] - snr_range[1])

    density = Float64(num_stations)
    collision_p = clamp(0.05 * num_stations, 0.0, 0.6)

    BandScene(contest, mode, propagation, snr_db,
              density, collision_p, stations)
end

"""
    noise_amplitude(scene) -> Float64

Compute linear noise RMS amplitude from the scene's SNR target.

Reference signal is the strongest station at median path loss:
    noise_amp = ref_received / 10^(snr_db / 20)
"""
function noise_amplitude(scene::BandScene)
    max_amp = maximum(s.signal_amplitude for s in scene.stations)
    ref_received = max_amp * 10.0^(-scene.propagation.path_loss_mean / 20.0)
    ref_received / 10.0^(scene.snr_db / 20.0)
end

# ── Operator style selection ─────────────────────────────────────────────────

const CONTEST_STYLE_POOL = AbstractOperatorStyle[
    FastContestOp(), MidSkillContestOp(), QRPOperator(), BeginnerOp()
]
const CONTEST_STYLE_WEIGHTS = [3.0, 3.0, 2.0, 0.5]

"""
    select_styles(rng, mode, n) -> Vector{AbstractOperatorStyle}

Select operator styles appropriate for the conversation mode.
"""
function select_styles(rng::AbstractRNG, ::ContestMode, n::Int)
    dist = Categorical(CONTEST_STYLE_WEIGHTS ./ sum(CONTEST_STYLE_WEIGHTS))
    [CONTEST_STYLE_POOL[rand(rng, dist)] for _ in 1:n]
end

function select_styles(rng::AbstractRNG, ::RagchewMode, n::Int)
    pool = [CasualRagchewer(), MidSkillContestOp(), QRPOperator(), BeginnerOp()]
    [pool[rand(rng, 1:length(pool))] for _ in 1:n]
end

function select_styles(rng::AbstractRNG, ::DXPileupMode, n::Int)
    styles = AbstractOperatorStyle[DXPileupManager()]
    callers = [FastContestOp(), MidSkillContestOp(), QRPOperator(), CasualRagchewer()]
    for _ in 2:n
        push!(styles, callers[rand(rng, 1:length(callers))])
    end
    styles
end

function select_styles(rng::AbstractRNG, ::TestMode, n::Int)
    pool = [BeginnerOp(), CasualRagchewer(), MidSkillContestOp()]
    [pool[rand(rng, 1:length(pool))] for _ in 1:n]
end

# ── Signal strength ──────────────────────────────────────────────────────────

"""
    signal_strength(rng, scene, station) -> Float64

Compute received signal strength for a station considering
propagation and station amplitude.
"""
function signal_strength(rng::AbstractRNG, scene::BandScene, station::Station)
    path_loss = scene.propagation.path_loss_mean + randn(rng) * scene.propagation.path_loss_sigma
    raw = station.signal_amplitude * 10.0^(-path_loss / 20.0)
    max(0.001, raw)
end
