"""
    MorseSimulator.jl/src/dataset.jl

High-level dataset generation API. Produces batches of
(mel_spectrogram, label) pairs for neural network training.
"""

using Random

"""
    DatasetSample{T<:AbstractFloat}

A single training sample.

# Fields
- `mel_spectrogram::Matrix{T}` — mel spectrogram
- `label::String` — text label
- `metadata::Dict{String,Any}` — additional metadata
- `token_timing::AbstractTokenTiming` — TokenTiming for exact chunk alignment, NoTiming() when
  alignment unavailable. Enables dispatch in downstream (e.g. MorseDecoder).
"""
struct DatasetSample{T<:AbstractFloat}
    mel_spectrogram::Matrix{T}
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
- `filterbank::MelFilterbank{Float64}` — mel filterbank
- `path::AbstractSpectrogramPath` — generation path (AudioPath or DirectPath)
- `num_stations_range::UnitRange{Int}` — range of station counts
"""
struct DatasetConfig
    sample_rate::Int
    stft_config::STFTConfig{Float64}
    filterbank::MelFilterbank{Float64}
    path::AbstractSpectrogramPath
    num_stations_range::UnitRange{Int}
end

"""
    DatasetConfig(; path, sample_rate, fft_size, hop_size, n_mels, f_min, f_max, stations) -> DatasetConfig

Create dataset configuration with defaults optimized for CW Morse.
"""
function DatasetConfig(;
        path::AbstractSpectrogramPath = DirectPath(),
        sample_rate::Int = 44100,
        fft_size::Int = 1024,
        hop_size::Int = 128,   # ~2.9 ms/frame @ 44.1 kHz → 4–5+ frames per dot
        n_mels::Int = 40,
        f_min::Float64 = 200.0,
        f_max::Float64 = 900.0,
        stations::UnitRange{Int} = 2:6)

    stft = STFTConfig(fft_size=fft_size, hop_size=hop_size)
    fb = MelFilterbank(n_filters=n_mels, fft_size=fft_size,
                       sample_rate=sample_rate, f_min=f_min, f_max=f_max)

    DatasetConfig(sample_rate, stft, fb, path, stations)
end

# ============================================================================
# Sample Generation
# ============================================================================

"""
    generate_sample(rng, config) -> DatasetSample
    generate_sample(rng, config, transcript, scene) -> DatasetSample

Generate a single training sample.

- **Without transcript**: draws a new band scene and generates a new transcript
  internally. Use a varying `rng` (e.g. different seeds or advancing the RNG)
  to get different content each time.

- **With transcript and scene**: uses the given transcript and band scene to
  generate the spectrogram (and optionally audio). Use this when you have
  already generated and inspected a transcript and want to produce the
  corresponding sample from it.
"""
function generate_sample(rng::AbstractRNG, config::DatasetConfig)
    n_stations = rand(rng, config.num_stations_range)

    # One scene: transcript and spectrogram must use the same stations/callsigns
    scene = BandScene(rng; num_stations=n_stations)
    transcript = generate_transcript(rng, scene)

    return generate_sample(rng, config, transcript, scene)
end

function generate_sample(rng::AbstractRNG, config::DatasetConfig,
                        transcript::Transcript, scene::BandScene)
    result, scene_events = generate_spec_and_events(config.path, rng, transcript, scene, config)

    n_stations = length(scene.stations)
    metadata = Dict{String,Any}(
        "mode" => transcript.mode_name,
        "contest" => transcript.contest_name,
        "num_stations" => n_stations,
        "duration" => result.duration,
        "sample_rate" => config.sample_rate
    )

    n_frames = size(result.mel_spectrogram, 2)
    hop_size = config.stft_config.hop_size
    token_timing = compute_token_timing(
        transcript, scene_events, n_frames, result.duration,
        config.sample_rate, hop_size,
    )

    return DatasetSample{Float64}(result.mel_spectrogram, result.label, metadata, token_timing)
end

# Dispatch: return (result, scene_events) for alignment; no leading underscore per style.
function generate_spec_and_events(::DirectPath, rng, transcript, scene, config)
    generate_spectrogram(DirectPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        filterbank=config.filterbank)
end

function generate_spec_and_events(::AudioPath, rng, transcript, scene, config)
    result, _, scene_events = generate_spectrogram(AudioPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        filterbank=config.filterbank)
    return result, scene_events
end

"""
    generate_dataset(rng, n, config) -> Vector{DatasetSample}

Generate `n` training samples.
"""
function generate_dataset(rng::AbstractRNG, n::Int, config::DatasetConfig)
    samples = Vector{DatasetSample{Float64}}(undef, n)
    for i in 1:n
        samples[i] = generate_sample(rng, config)
    end
    return samples
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

- **Without transcript**: generates a new scene and transcript internally; vary
  `rng` for different content.
- **With transcript and scene**: uses the given transcript and scene to generate
  the audio and spectrogram (e.g. after inspecting the transcript).
"""
function generate_sample_with_audio(rng::AbstractRNG, config::DatasetConfig)
    n_stations = rand(rng, config.num_stations_range)
    scene = BandScene(rng; num_stations=n_stations)
    transcript = generate_transcript(rng, scene)
    return generate_sample_with_audio(rng, config, transcript, scene)
end

function generate_sample_with_audio(rng::AbstractRNG, config::DatasetConfig,
                                   transcript::Transcript, scene::BandScene)
    result, mixed, scene_events = generate_spectrogram(AudioPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        filterbank=config.filterbank)

    n_stations = length(scene.stations)
    metadata = Dict{String,Any}(
        "mode" => transcript.mode_name,
        "contest" => transcript.contest_name,
        "num_stations" => n_stations,
        "duration" => result.duration,
        "sample_rate" => config.sample_rate
    )

    n_frames = size(result.mel_spectrogram, 2)
    hop_size = config.stft_config.hop_size
    token_timing = compute_token_timing(
        transcript, scene_events, n_frames, result.duration,
        config.sample_rate, hop_size,
    )
    sample = DatasetSample{Float64}(result.mel_spectrogram, result.label, metadata, token_timing)
    return sample, mixed
end

generate_sample_with_audio(config::DatasetConfig) =
    generate_sample_with_audio(Random.default_rng(), config)
