"""
    CWContestSim/src/spectrogram/mel.jl

Mel-scale filterbank construction for spectrogram generation.
Optimized for CW Morse signals (200-900 Hz range).
"""

"""
    MelFilterbank{T<:AbstractFloat}

Triangular mel-scale filterbank.

# Fields
- `filters::Matrix{T}` — filter weights (n_filters × n_fft_bins)
- `n_filters::Int` — number of mel filters
- `fft_size::Int` — FFT size
- `sample_rate::Int` — sample rate
- `f_min::T` — minimum frequency (Hz)
- `f_max::T` — maximum frequency (Hz)
"""
struct MelFilterbank{T<:AbstractFloat}
    filters::Matrix{T}
    n_filters::Int
    fft_size::Int
    sample_rate::Int
    f_min::T
    f_max::T
end

"""
    hz_to_mel(f) -> Float64

Convert frequency in Hz to mel scale.
"""
hz_to_mel(f::Real) = 2595.0 * log10(1.0 + Float64(f) / 700.0)

"""
    mel_to_hz(m) -> Float64

Convert mel scale to frequency in Hz.
"""
mel_to_hz(m::Real) = 700.0 * (10.0^(Float64(m) / 2595.0) - 1.0)

"""
    MelFilterbank(; n_filters, fft_size, sample_rate, f_min, f_max) -> MelFilterbank

Construct a mel filterbank with triangular filters.

Default configuration is optimized for CW Morse signals.
"""
function MelFilterbank(;
        n_filters::Int = 40,
        fft_size::Int = 1024,
        sample_rate::Int = 44100,
        f_min::Float64 = 200.0,
        f_max::Float64 = 900.0)

    T = Float64

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Mel-spaced center frequencies
    mel_points = range(mel_min, mel_max, length=n_filters + 2)
    hz_points = mel_to_hz.(mel_points)

    # FFT bin frequencies
    n_bins = fft_size ÷ 2 + 1
    fft_freqs = range(zero(T), T(sample_rate) / 2, length=n_bins)

    # Build triangular filters
    filters = zeros(T, n_filters, n_bins)

    for m in 1:n_filters
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        for k in 1:n_bins
            f = fft_freqs[k]
            if f >= f_left && f <= f_center && f_center > f_left
                filters[m, k] = (f - f_left) / (f_center - f_left)
            elseif f > f_center && f <= f_right && f_right > f_center
                filters[m, k] = (f_right - f) / (f_right - f_center)
            end
        end
    end

    return MelFilterbank{T}(filters, n_filters, fft_size, sample_rate,
                            T(f_min), T(f_max))
end

"""
    apply_filterbank(fb, power_spectrum) -> Vector{T}

Apply mel filterbank to a power spectrum, return mel energies.
"""
function apply_filterbank(fb::MelFilterbank{T},
                          power_spectrum::AbstractVector{T}) where T<:AbstractFloat
    n_bins = min(size(fb.filters, 2), length(power_spectrum))
    mel_energies = zeros(T, fb.n_filters)

    @inbounds for m in 1:fb.n_filters
        e = zero(T)
        for k in 1:n_bins
            e += fb.filters[m, k] * power_spectrum[k]
        end
        mel_energies[m] = e
    end

    return mel_energies
end

"""
    apply_filterbank(fb, spectrogram_matrix) -> Matrix{T}

Apply mel filterbank to an entire spectrogram (freq × time).
Returns mel_spectrogram (n_filters × time).
"""
function apply_filterbank(fb::MelFilterbank{T},
                          spec::AbstractMatrix{T}) where T<:AbstractFloat
    n_frames = size(spec, 2)
    mel_spec = zeros(T, fb.n_filters, n_frames)

    @inbounds for j in 1:n_frames
        n_bins = min(size(fb.filters, 2), size(spec, 1))
        for m in 1:fb.n_filters
            e = zero(T)
            for k in 1:n_bins
                e += fb.filters[m, k] * spec[k, j]
            end
            mel_spec[m, j] = e
        end
    end

    return mel_spec
end
