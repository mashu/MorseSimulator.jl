# MorseSimulator.jl

A Julia package for simulating realistic **CW Morse amateur radio communications** and generating **mel-spectrogram datasets** for training Whisper-like audio-to-text neural networks.

## Architecture

The simulation pipeline consists of five layers:

```
Conversation Layer → Text Layer → Morse Timing Layer → Signal Layer → Spectrogram Layer
```

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ Conversation Layer                                                  │
│  BandScene + Stations + Contest → Transcript (ordered transmissions)│
│  Modes: Contest | Ragchew | DX Pileup | Test                       │
├─────────────────────────────────────────────────────────────────────┤
│ Text Layer                                                          │
│  Transcript → "<START> CQ CQ DE SO5KM ... <END>"                   │
│  Labels for neural network training                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Morse Timing Layer                                                  │
│  Text → TimedMorseEvents (dots, dashes, gaps with jitter)           │
│  Operator speed model: WPM ~ Normal(μ, σ) with drift               │
├─────────────────────────────────────────────────────────────────────┤
│ Signal Layer                                                        │
│  Events → Envelope → CW Tone → Channel Effects → Mixed Signal      │
│  QSB fading, noise, frequency offset, multi-station mixing          │
├─────────────────────────────────────────────────────────────────────┤
│ Spectrogram Layer                                                   │
│  Mode 1 (Audio Path):  Signal → STFT → Mel Filterbank → Spectrogram│
│  Mode 2 (Direct Path): Events → Analytic Mel Spectrogram           │
│  Consistency metrics between paths                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/MorseSimulator.jl")
```

## Quick Start

### Phase 1: Generate Transcripts

```julia
using MorseSimulator, Random

rng = MersenneTwister(42)

# Single transcript
transcript = generate_transcript(rng; num_stations=3, mode=ContestMode())
plot_transcript(transcript)

# Batch generation
transcripts = generate_transcripts(rng, 100)

# Inspect the training label
println(flat_text(transcript))
# <START> CQ CQ SO5KM SO5KM DE DL3AB 5NN 001 TU DL3AB 5NN 002 TU SO5KM <END>

# Annotated text with TX markers
println(annotated_text(transcript))
```

### Phase 2: Generate Spectrograms

```julia
# Fast dataset generation (no audio)
config = DatasetConfig(path=DirectPath(), n_mels=40, f_min=200.0, f_max=900.0)
sample = generate_sample(rng, config)
display(plot_spectrogram(sample.mel_spectrogram))

# Generate with audio for inspection
sample, audio = generate_sample_with_audio(rng, config)
save_audio("inspection.wav", audio)
display(plot_waveform(audio))

# Batch dataset
dataset = generate_dataset(rng, 1000, config)
```

### Compare Generation Paths

```julia
transcript = generate_transcript(rng; num_stations=2)
scene = BandScene(rng; num_stations=2)

spec_audio, _ = generate_spectrogram(AudioPath(), rng, transcript, scene)
spec_direct = generate_spectrogram(DirectPath(), rng, transcript, scene)

report = compare_paths(spec_audio, spec_direct)
plot_consistency(report)
```

## Design Principles

### Type Hierarchy and Dispatch

The package uses Julia's type system extensively:

```
AbstractContest          → CQWWContest, CQWPXContest, ARRLDXContest, ...
AbstractOperatorStyle    → FastContestOp, CasualRagchewer, BeginnerOp, ...
AbstractConversationMode → ContestMode, RagchewMode, DXPileupMode, TestMode
AbstractMorseElement     → Dot, Dash, SymbolGap, CharGap, WordGap
AbstractEnvelope{T}      → RaisedCosineEnvelope{T}, HardEnvelope{T}
AbstractNoiseModel{T}    → GaussianNoise{T}, ImpulsiveNoise{T}
AbstractFadingModel{T}   → SinusoidalFading{T}, RayleighFading{T}
AbstractSpectrogramPath  → AudioPath, DirectPath
AbstractConsistencyMetric→ L2SpectralError, CosineSimilarity, KLDivergence, ...
```

Behavior differences are expressed via **multiple dispatch**, not `if`/`typeof`/`isa` checks.
For example, the same function `generate_cq` produces different CQ calls depending on operator style:

```julia
generate_cq(rng, station, ::FastContestOp)      # "CQ SO5KM"
generate_cq(rng, station, ::CasualRagchewer)     # "CQ CQ CQ DE SO5KM SO5KM K"
generate_cq(rng, station, ::DXPileupManager)     # "CQ DX SO5KM"
```

### Parametric Types

Signal-layer types are parameterized on floating-point precision:

```julia
struct RaisedCosineEnvelope{T<:AbstractFloat} <: AbstractEnvelope{T}
    rise_time::T
end
```

### Stochastic Models

All randomness flows through explicit `AbstractRNG` arguments for reproducibility:

```julia
rng = MersenneTwister(42)
transcript = generate_transcript(rng; num_stations=4)
```

## Package Structure

```
MorseSimulator.jl/
├── Project.toml
├── src/
│   ├── MorseSimulator.jl         # Main module
│   ├── types.jl                   # Abstract type hierarchy
│   ├── dataset.jl                 # High-level dataset API
│   ├── transcript/
│   │   ├── callsigns.jl          # Callsign generation
│   │   ├── contests.jl           # Contest types & exchanges
│   │   ├── operators.jl          # Operator styles & state machine
│   │   ├── stations.jl           # Station model
│   │   ├── phrases.jl            # Phrase templates & conversation modes
│   │   ├── transmission.jl       # Transmission record type
│   │   ├── bandscene.jl          # Band environment model
│   │   └── conversation.jl       # Conversation engine
│   ├── morse/
│   │   ├── code.jl               # ITU Morse code table
│   │   ├── timing.jl             # WPM → timing with jitter
│   │   └── encoder.jl            # Transcript → timed events
│   ├── signal/
│   │   ├── envelope.jl           # Keying envelope shapes
│   │   ├── tone.jl               # CW tone generation
│   │   ├── channel.jl            # Noise & fading models
│   │   ├── mixer.jl              # Multi-station mixing
│   │   └── audio.jl              # WAV file I/O
│   ├── spectrogram/
│   │   ├── mel.jl                # Mel filterbank
│   │   ├── stft.jl               # STFT computation
│   │   ├── audio_path.jl         # Mode 1: via waveform
│   │   ├── direct_path.jl        # Mode 2: analytic
│   │   └── consistency.jl        # Path comparison metrics
│   └── visualization/
│       └── plotting.jl           # UnicodePlots visualization
├── test/
│   └── runtests.jl
└── docs/
```

## Contest Types Supported

| Contest | Exchange |
|---------|----------|
| CQ WW DX | RST + CQ Zone |
| CQ WPX | RST + Serial |
| ARRL DX | RST + State/Power |
| IARU HF | RST + ITU Zone |
| SP DX | RST + Province/Serial |
| WAE | RST + Serial |
| All Asian DX | RST + Age |
| Generic Serial | RST + Serial |

## Operator Styles

| Style | Mean WPM | Verbosity | Patience |
|-------|----------|-----------|----------|
| FastContestOp | 32 | 5% | 2 retries |
| CasualRagchewer | 20 | 85% | 5 retries |
| BeginnerOp | 12 | 60% | 4 retries |
| DXPileupManager | 28 | 2% | 1 retry |
| MidSkillContestOp | 24 | 20% | 3 retries |
| QRPOperator | 18 | 30% | 6 retries |

## Signal Parameters

- **CW tone**: 400–900 Hz (configurable per station)
- **Noise**: Gaussian + optional impulsive (QRN)
- **Fading**: Sinusoidal QSB (0.1–1 Hz) or Rayleigh
- **Frequency offset**: per-station offset ~Normal(0, 5 Hz)
- **Mel spectrogram range**: 200–900 Hz (40 mel bins default)
- **Audio format**: PCM 16-bit, 44100 Hz, Mono

## Dependencies

- `Distributions.jl` — probabilistic models
- `FFTW.jl` — fast Fourier transforms
- `DSP.jl` — signal processing utilities
- `LibSndFile.jl` — WAV audio I/O
- `UnicodePlots.jl` — terminal visualization

## License

MIT
