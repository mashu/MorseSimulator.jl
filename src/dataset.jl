"""
    MorseSimulator.jl/src/dataset.jl

High-level dataset generation API. Produces batches of
(spectrogram, label) pairs for neural network training.
Uses linear band (not mel) for ~10 Hz resolution in 200–900 Hz.
"""

using Random

"""
    DatasetSample{T<:AbstractFloat}

A single training sample.

# Fields
- `spectrogram::Matrix{T}` — linear band spectrogram (n_bins × n_frames)
- `label::String` — text label
- `metadata::Dict{String,Any}` — additional metadata
- `token_timing::AbstractTokenTiming` — TokenTiming for exact chunk alignment, NoTiming() when
  alignment unavailable. Enables dispatch in downstream (e.g. MorseDecoder).
"""
struct DatasetSample{T<:AbstractFloat}
    spectrogram::Matrix{T}
    label::String
    metadata::Dict{String,Any}
    token_timing::AbstractTokenTiming
end

"""
    DatasetConfig

Configuration for dataset generation.

# Fields
- `sample_rate::Int` — audio sample rate
- `stft_config::STFTConfig{Float64}` — STFT parameters
- `linear_band::LinearBand` — frequency band (linear FFT bins, ~10 Hz resolution)
- `path::AbstractSpectrogramPath` — generation path (AudioPath or DirectPath)
- `num_stations_range::UnitRange{Int}` — range of station counts
"""
struct DatasetConfig
    sample_rate::Int
    stft_config::STFTConfig{Float64}
    linear_band::LinearBand
    path::AbstractSpectrogramPath
    num_stations_range::UnitRange{Int}
end

"""
    DatasetConfig(; path, sample_rate, fft_size, hop_size, f_min, f_max, stations) -> DatasetConfig

Create dataset configuration. Uses linear band (no mel) for CW: ~(sr/fft_size) Hz per bin.
Default 4096 FFT @ 44.1 kHz → ~10.77 Hz resolution in 200–900 Hz.
"""
function DatasetConfig(;
        path::AbstractSpectrogramPath = DirectPath(),
        sample_rate::Int = 44100,
        fft_size::Int = 4096,
        hop_size::Int = 128,
        f_min::Float64 = 200.0,
        f_max::Float64 = 900.0,
        stations::UnitRange{Int} = 2:6)

    stft = STFTConfig(fft_size=fft_size, hop_size=hop_size)
    lb = LinearBand(fft_size=fft_size, sample_rate=sample_rate, f_min=f_min, f_max=f_max)
    DatasetConfig(sample_rate, stft, lb, path, stations)
end

# ============================================================================
# Shared metadata construction
# ============================================================================

function sample_metadata(transcript::Transcript, scene::BandScene, sample_rate::Int, duration::Float64)
    Dict{String,Any}(
        "mode"         => transcript.mode_name,
        "contest"      => transcript.contest_name,
        "num_stations" => length(scene.stations),
        "duration"     => duration,
        "sample_rate"  => sample_rate,
        "snr_db"       => scene.snr_db,
    )
end

# ============================================================================
# Sample Generation
# ============================================================================

"""
    generate_sample(rng, config) -> DatasetSample
    generate_sample(rng, config, transcript, scene) -> DatasetSample

Generate a single training sample.

- **Without transcript**: draws a new band scene and generates a new transcript
  internally. Use a varying `rng` for different content each time.

- **With transcript and scene**: uses the given transcript and band scene to
  generate the spectrogram. Use when you have already inspected a transcript.
"""
function generate_sample(rng::AbstractRNG, config::DatasetConfig)
    n_stations = rand(rng, config.num_stations_range)
    scene = BandScene(rng; num_stations=n_stations)
    transcript = generate_transcript(rng, scene)
    generate_sample(rng, config, transcript, scene)
end

function generate_sample(rng::AbstractRNG, config::DatasetConfig,
                        transcript::Transcript, scene::BandScene)
    result, scene_events = generate_spec_and_events(config.path, rng, transcript, scene, config)

    metadata = sample_metadata(transcript, scene, config.sample_rate, result.duration)

    n_frames = size(result.spectrogram, 2)
    token_timing = compute_token_timing(
        transcript, scene_events, n_frames, result.duration,
        config.sample_rate, config.stft_config.hop_size,
    )

    DatasetSample{Float64}(result.spectrogram, result.label, metadata, token_timing)
end

# Dispatch on path type for spectrogram generation
function generate_spec_and_events(::DirectPath, rng, transcript, scene, config)
    generate_spectrogram(DirectPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        linear_band=config.linear_band)
end

function generate_spec_and_events(::AudioPath, rng, transcript, scene, config)
    result, _, scene_events = generate_spectrogram(AudioPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        linear_band=config.linear_band)
    return result, scene_events
end

"""
    generate_dataset(rng, n, config) -> Vector{DatasetSample}

Generate `n` training samples.
"""
function generate_dataset(rng::AbstractRNG, n::Int, config::DatasetConfig)
    [generate_sample(rng, config) for _ in 1:n]
end

generate_dataset(n::Int, config::DatasetConfig) =
    generate_dataset(Random.default_rng(), n, config)

generate_dataset(n::Int; kwargs...) =
    generate_dataset(Random.default_rng(), n, DatasetConfig(; kwargs...))

# ============================================================================
# Sample with Audio (for inspection)
# ============================================================================

"""
    generate_sample_with_audio(rng, config) -> (DatasetSample, MixedSignal)
    generate_sample_with_audio(rng, config, transcript, scene) -> (DatasetSample, MixedSignal)

Generate a sample with both spectrogram and audio for inspection.
Always uses AudioPath (full audio synthesis) regardless of config.
"""
function generate_sample_with_audio(rng::AbstractRNG, config::DatasetConfig)
    n_stations = rand(rng, config.num_stations_range)
    scene = BandScene(rng; num_stations=n_stations)
    transcript = generate_transcript(rng, scene)
    generate_sample_with_audio(rng, config, transcript, scene)
end

function generate_sample_with_audio(rng::AbstractRNG, config::DatasetConfig,
                                   transcript::Transcript, scene::BandScene)
    result, mixed, scene_events = generate_spectrogram(AudioPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        linear_band=config.linear_band)

    metadata = sample_metadata(transcript, scene, config.sample_rate, result.duration)

    n_frames = size(result.spectrogram, 2)
    token_timing = compute_token_timing(
        transcript, scene_events, n_frames, result.duration,
        config.sample_rate, config.stft_config.hop_size,
    )

    sample = DatasetSample{Float64}(result.spectrogram, result.label, metadata, token_timing)
    return sample, mixed
end

generate_sample_with_audio(config::DatasetConfig) =
    generate_sample_with_audio(Random.default_rng(), config)
