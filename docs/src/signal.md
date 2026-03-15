# Signal & Spectrogram

## Morse timing

Text is converted to timed Morse events (`TimedMorseEvent`) with realistic operator jitter and WPM drift. The `TimingParams` struct holds dot duration, jitter sigma, and drift rate derived from the target WPM.

Each event stores a `MorseElement` (the five-type union `Dot | Dash | SymbolGap | CharGap | WordGap`), a start time, and an actual duration.

## Keying envelope

Two envelope shapes are available via dispatch on `AbstractEnvelope{T}`:

- `RaisedCosineEnvelope{T}` — smooth rise/fall edges to reduce key clicks (default).
- `HardEnvelope{T}` — instantaneous on/off transitions.

## Channel effects

### Noise models

- `GaussianNoise{T}(power)` — additive white Gaussian noise.
- `ImpulsiveNoise{T}(power, rate)` — burst noise (QRN).

### Fading models

- `SinusoidalFading{T}(depth, rate, phase)` — regular QSB.
- `RayleighFading{T}(scale)` — random deep fades.

All effects are applied in-place via `apply_noise!`, `apply_fading!`, and `apply_channel!`.

## Multi-station mixing

`mix_signals` combines per-station CW signals (each at its own tone frequency and amplitude) into a single mixed waveform. The result is a `MixedSignal` containing the composite audio samples and metadata.

## Mel spectrogram

Two generation paths produce equivalent mel spectrograms:

| Path | Method | Use case |
|------|--------|----------|
| `DirectPath()` | Analytic: events → mel power directly | Training (fast) |
| `AudioPath()` | Waveform → STFT → mel filterbank | Inspection, WAV export |

Both paths apply peak normalization and log₁₀ compression. The `DirectPath` includes the same variations as the audio path: per-station amplitude, noise floor, overlap, and QSB fading.

## Token alignment

The alignment module (`alignment.jl`) maps each token in the training label to spectrogram frame ranges using the simulator's timing data. This produces `TokenTiming` (or `NoTiming` when alignment fails), which downstream code uses via dispatch to slice spectrograms into correctly-labelled chunks for training.
