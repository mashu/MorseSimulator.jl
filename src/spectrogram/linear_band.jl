"""
    MorseSimulator.jl/src/spectrogram/linear_band.jl

Linear-frequency band for spectrograms. Uses raw FFT bins in [f_min, f_max].
Bin width = sample_rate/fft_size Hz. For 8 Hz or better separation use target_bin_width_Hz=8
(e.g. fft_size=8192 @ 44.1 kHz → ~5.4 Hz/bin). Time resolution is unchanged (set by hop_size).
"""

"""
    LinearBand

Configuration for a linear-frequency band (slice of FFT bins).
No mel compression — one output bin per FFT bin in the band.

# Fields
- `fft_size::Int`
- `sample_rate::Int`
- `f_min::Float64` — band low edge (Hz)
- `f_max::Float64` — band high edge (Hz)
- `bin_lo::Int` — first FFT bin index (1-based) in band
- `bin_hi::Int` — last FFT bin index (1-based) in band
"""
struct LinearBand
    fft_size::Int
    sample_rate::Int
    f_min::Float64
    f_max::Float64
    bin_lo::Int
    bin_hi::Int
end

"""
    LinearBand(; fft_size, sample_rate, f_min, f_max) -> LinearBand
    LinearBand(; sample_rate, f_min, f_max, target_bin_width_Hz, n_bins) -> LinearBand

Construct a linear band. Either pass fft_size explicitly, or pass target_bin_width_Hz (e.g. 8)
and optionally n_bins (e.g. 128). For the latter, fft_size is chosen so bin width ≤ target
(smallest power-of-2 ≥ sample_rate/target_bin_width_Hz). If n_bins is given, f_max is set so
the band has exactly that many bins (f_max = f_min + n_bins * bin_width, then bin_hi trimmed).
"""
function LinearBand(;
    fft_size::Union{Int,Nothing} = nothing,
    sample_rate::Int = 44100,
    f_min::Float64 = 200.0,
    f_max::Union{Float64,Nothing} = 900.0,
    target_bin_width_Hz::Union{Float64,Nothing} = nothing,
    n_bins::Union{Int,Nothing} = nothing)
    if fft_size === nothing
        if target_bin_width_Hz !== nothing || n_bins !== nothing
            # New API: choose fft_size for target resolution (8 Hz or better)
            target = something(target_bin_width_Hz, 10.0)
            min_fft = ceil(Int, sample_rate / target)
            fft_size = nextpow(2, min_fft)
        else
            fft_size = 4096  # original default
        end
    end
    n_fft_bins = fft_size ÷ 2 + 1
    bin_lo = max(1, 1 + floor(Int, f_min * fft_size / sample_rate))
    if n_bins !== nothing
        bin_hi = min(n_fft_bins, bin_lo + n_bins - 1)
    elseif f_max !== nothing
        bin_hi = min(n_fft_bins, 1 + floor(Int, f_max * fft_size / sample_rate))
    else
        bin_hi = n_fft_bins
    end
    bin_lo = min(bin_lo, bin_hi)
    actual_f_max = (bin_hi - 1) * sample_rate / fft_size
    LinearBand(fft_size, sample_rate, f_min, actual_f_max, bin_lo, bin_hi)
end

"""Number of frequency bins in the band."""
n_bins(lb::LinearBand) = max(0, lb.bin_hi - lb.bin_lo + 1)

"""1-based FFT bin index for frequency f (Hz)."""
function fft_bin_for_freq(lb::LinearBand, f::Real)
    k = 1 + round(Int, Float64(f) * lb.fft_size / lb.sample_rate)
    clamp(k, 1, lb.fft_size ÷ 2 + 1)
end

"""Output band bin index (1 to n_bins) for 1-based FFT bin k, or 0 if outside band."""
function band_bin_for_fft_bin(lb::LinearBand, k::Int)
    if k < lb.bin_lo || k > lb.bin_hi
        return 0
    end
    k - lb.bin_lo + 1
end

"""Center frequency (Hz) of band bin b (1-based)."""
function bin_center_hz(lb::LinearBand, b::Int)
    k = lb.bin_lo + b - 1
    (k - 1) * lb.sample_rate / lb.fft_size
end
