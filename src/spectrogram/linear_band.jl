"""
    MorseSimulator.jl/src/spectrogram/linear_band.jl

Linear-frequency band for spectrograms. Uses raw FFT bins in [f_min, f_max].
Gives ~(sr/fft_size) Hz resolution (e.g. ~10.77 Hz at 44.1 kHz, fft_size=4096).
Better suited than mel for narrow-band CW where we need to separate stations 10 Hz apart.
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

function LinearBand(;
    fft_size::Int = 4096,
    sample_rate::Int = 44100,
    f_min::Float64 = 200.0,
    f_max::Float64 = 900.0)
    n_fft_bins = fft_size ÷ 2 + 1
    bin_lo = max(1, 1 + floor(Int, f_min * fft_size / sample_rate))
    bin_hi = min(n_fft_bins, 1 + floor(Int, f_max * fft_size / sample_rate))
    bin_lo = min(bin_lo, bin_hi)  # ensure valid range
    LinearBand(fft_size, sample_rate, f_min, f_max, bin_lo, bin_hi)
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
