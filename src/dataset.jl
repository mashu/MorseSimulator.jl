"""
    CWContestSim/src/dataset.jl

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
"""
struct DatasetSample{T<:AbstractFloat}
    mel_spectrogram::Matrix{T}
    label::String
    metadata::Dict{String,Any}
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
        hop_size::Int = 256,
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

Generate a single training sample.
"""
function generate_sample(rng::AbstractRNG, config::DatasetConfig)
    n_stations = rand(rng, config.num_stations_range)

    # Generate transcript
    transcript = generate_transcript(rng; num_stations=n_stations)

    # Generate scene for signal parameters
    scene = BandScene(rng; num_stations=n_stations)

    # Generate spectrogram
    result = _generate_spec(config.path, rng, transcript, scene, config)

    metadata = Dict{String,Any}(
        "mode" => transcript.mode_name,
        "contest" => transcript.contest_name,
        "num_stations" => n_stations,
        "duration" => result.duration,
        "sample_rate" => config.sample_rate
    )

    return DatasetSample{Float64}(result.mel_spectrogram, result.label, metadata)
end

# Dispatch helper to handle return types
function _generate_spec(::DirectPath, rng, transcript, scene, config)
    generate_spectrogram(DirectPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        filterbank=config.filterbank)
end

function _generate_spec(::AudioPath, rng, transcript, scene, config)
    result, _ = generate_spectrogram(AudioPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        filterbank=config.filterbank)
    return result
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

Generate a sample with both spectrogram and audio for inspection.
Always uses AudioPath regardless of config.
"""
function generate_sample_with_audio(rng::AbstractRNG, config::DatasetConfig)
    n_stations = rand(rng, config.num_stations_range)
    transcript = generate_transcript(rng; num_stations=n_stations)
    scene = BandScene(rng; num_stations=n_stations)

    result, mixed = generate_spectrogram(AudioPath(), rng, transcript, scene;
        sample_rate=config.sample_rate,
        stft_config=config.stft_config,
        filterbank=config.filterbank)

    metadata = Dict{String,Any}(
        "mode" => transcript.mode_name,
        "contest" => transcript.contest_name,
        "num_stations" => n_stations,
        "duration" => result.duration,
        "sample_rate" => config.sample_rate
    )

    sample = DatasetSample{Float64}(result.mel_spectrogram, result.label, metadata)
    return sample, mixed
end

generate_sample_with_audio(config::DatasetConfig) =
    generate_sample_with_audio(Random.default_rng(), config)
