"""
    MorseSimulator.jl/src/spectrogram/stft.jl

Short-Time Fourier Transform computation using FFTW.
Provides the power spectrogram needed for mel conversion.
"""

using FFTW, DSP

"""
    STFTConfig{T<:AbstractFloat}

STFT configuration parameters.

# Fields
- `fft_size::Int` — FFT window size
- `hop_size::Int` — hop between frames
- `window::Vector{T}` — window function
"""
struct STFTConfig{T<:AbstractFloat}
    fft_size::Int
    hop_size::Int
    window::Vector{T}
end

"""
    STFTConfig(; fft_size, hop_size, window_type) -> STFTConfig

Create STFT configuration. Default: Hann window, 1024 FFT, 256 hop.
"""
function STFTConfig(;
        fft_size::Int = 1024,
        hop_size::Int = 256,
        window_type::Symbol = :hann)

    T = Float64
    win = _make_window(Val(window_type), fft_size, T)
    STFTConfig{T}(fft_size, hop_size, win)
end

# Window generation via dispatch on Val
_make_window(::Val{:hann}, n::Int, ::Type{T}) where T =
    T[T(0.5) * (one(T) - cos(T(2π) * i / (n - 1))) for i in 0:n-1]

_make_window(::Val{:hamming}, n::Int, ::Type{T}) where T =
    T[T(0.54) - T(0.46) * cos(T(2π) * i / (n - 1)) for i in 0:n-1]

_make_window(::Val{:rectangular}, n::Int, ::Type{T}) where T =
    ones(T, n)

"""
    compute_stft(signal, config) -> Matrix{ComplexF64}

Compute the STFT of a signal.
Returns complex spectrogram (n_freq × n_frames).
"""
function compute_stft(signal::Vector{T}, config::STFTConfig{T}) where T<:AbstractFloat
    n = length(signal)
    n_frames = max(1, (n - config.fft_size) ÷ config.hop_size + 1)
    n_freq = config.fft_size ÷ 2 + 1

    result = Matrix{Complex{T}}(undef, n_freq, n_frames)
    frame = Vector{T}(undef, config.fft_size)

    plan = FFTW.plan_rfft(frame)

    for j in 1:n_frames
        start_idx = (j - 1) * config.hop_size + 1
        end_idx = start_idx + config.fft_size - 1

        # Extract and window the frame
        @inbounds for i in 1:config.fft_size
            si = start_idx + i - 1
            if si <= n
                frame[i] = signal[si] * config.window[i]
            else
                frame[i] = zero(T)
            end
        end

        # FFT
        fft_result = plan * frame
        @inbounds for i in 1:n_freq
            result[i, j] = fft_result[i]
        end
    end

    return result
end

"""
    power_spectrogram(stft_result) -> Matrix{T}

Compute power spectrogram from complex STFT.
"""
function power_spectrogram(stft_result::Matrix{Complex{T}}) where T<:AbstractFloat
    return abs2.(stft_result)
end

"""
    log_power_spectrogram(stft_result; floor) -> Matrix{T}

Compute log-power spectrogram with floor value.
"""
function log_power_spectrogram(stft_result::Matrix{Complex{T}};
                                floor::T = T(1e-10)) where T<:AbstractFloat
    pspec = power_spectrogram(stft_result)
    return log10.(max.(pspec, floor))
end
