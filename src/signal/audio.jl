"""
    MorseSimulator.jl/src/signal/audio.jl

Audio file I/O for saving and loading WAV files.
Supports PCM 16-bit, 44100 Hz, Mono format.
Uses FileIO + LibSndFile for robust audio I/O.
"""

using FileIO
using LibSndFile
using SampledSignals: PCM16Sample

"""
    AudioConfig

Audio format configuration.

# Fields
- `sample_rate::Int` — samples per second (default 44100)
- `bit_depth::Int` — bits per sample (default 16)
- `channels::Int` — number of channels (default 1, mono)
"""
struct AudioConfig
    sample_rate::Int
    bit_depth::Int
    channels::Int
end

AudioConfig() = AudioConfig(44100, 16, 1)

"""
    save_audio(path, signal; config)

Save a mixed signal to a WAV file.
"""
function save_audio(path::AbstractString, signal::MixedSignal{T};
                    config::AudioConfig = AudioConfig()) where T<:AbstractFloat
    save_wav(path, signal.samples, signal.sample_rate)
end

"""
    save_wav(path, samples, sample_rate)

Save raw samples to a WAV file using FileIO + LibSndFile.
Output format: 44100 Hz, 16-bit PCM, mono (when using default config).
Samples should be in [-1, 1] range.
"""
function save_wav(path::AbstractString, samples::Vector{T},
                  sample_rate::Int) where T<:AbstractFloat
    # Ensure samples are in proper range, then convert to 16-bit PCM for compatibility (e.g. aplay)
    clipped = clamp.(samples, T(-1), T(1))
    data_16 = reshape(PCM16Sample.(clipped), :, 1)
    FileIO.save(path, data_16; samplerate=sample_rate)
    return path
end

"""
    load_wav(path) -> (samples::Vector{Float64}, sample_rate::Int)

Load a WAV file and return samples and sample rate.
"""
function load_wav(path::AbstractString)
    data = FileIO.load(path)
    sr = Int(LibSndFile.samplerate(data))
    # Convert to vector (mono)
    samples = Float64.(data[:, 1])
    return samples, sr
end
