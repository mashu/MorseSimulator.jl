# Phase 2: Signal & Spectrogram

## Morse Timing

Characters are converted to dots and dashes per ITU Morse Code.
Standard timing (PARIS method):

| Element | Duration |
|---------|----------|
| Dot | 1 unit |
| Dash | 3 units |
| Symbol gap | 1 unit |
| Character gap | 3 units |
| Word gap | 7 units |

Unit duration: `1.2 / WPM` seconds.

Operator speed varies: `WPM ~ Normal(mean_wpm, σ)` with drift.

## Signal Generation

Each station produces: `s(t) = A(t) · envelope(t) · sin(2πft)`

where:
- `A(t)` is the amplitude with QSB fading
- `envelope(t)` is the raised-cosine shaped Morse keying
- `f` is the station's CW tone frequency (400–900 Hz)

## Channel Effects

- **Gaussian noise**: additive white noise at configurable SNR
- **Impulsive noise (QRN)**: occasional burst noise
- **QSB fading**: sinusoidal or Rayleigh fading (0.1–1 Hz)
- **Frequency offset**: per-station Δf ~ Normal(0, 5 Hz)

## Multi-Station Mixing

Signals from all stations are summed linearly.
Overlapping transmissions create realistic collisions.

## Spectrogram Generation

### Mode 1: Audio Path

```
Morse events → Envelope → CW Tone → Channel → STFT → Mel Filterbank → Spectrogram
```

Produces audio that can be saved as WAV for human inspection.

### Mode 2: Direct Path (Fast)

```
Morse events → Analytic envelope at frame rate → Mel energy at tone frequency → Spectrogram
```

Bypasses waveform synthesis. Much faster for dataset generation.

### Consistency Metrics

Compare the two paths using:
- **L2 Spectral Error**: RMS difference
- **Cosine Similarity**: angular similarity
- **KL Divergence**: distributional difference
- **Mean Absolute Error**: average difference

```julia
report = compare_paths(spec_audio, spec_direct)
plot_consistency(report)
```
