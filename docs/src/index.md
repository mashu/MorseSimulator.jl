# MorseSimulator.jl

Simulate realistic CW Morse amateur radio communications and generate mel-spectrogram datasets for training neural network decoders.

## Pipeline

```
Conversation → Morse Timing → Signal → Mel Spectrogram
```

Each layer is built on Julia's multiple dispatch, allowing clean composition of contest formats, operator styles, propagation effects, and noise models.

## Quick start

```julia
using MorseSimulator, Random

rng = MersenneTwister(42)
config = DatasetConfig(path=DirectPath())

# One sample
sample = generate_sample(rng, config)
println(sample.label)

# With audio
sample, audio = generate_sample_with_audio(rng, config)
save_audio("test.wav", audio)

# Batch
dataset = generate_dataset(rng, 100, config)
```

## Inspect before generating

```julia
scene = BandScene(rng; num_stations=3)
transcript = generate_transcript(rng, scene)
plot_transcript(transcript)

# Generate from this exact transcript
sample = generate_sample(rng, config, transcript, scene)
```

## Two spectrogram paths

Both paths produce equivalent mel spectrograms:

- **`DirectPath()`** — analytic; builds the mel spectrogram from Morse events without audio synthesis. Fast, used for training.
- **`AudioPath()`** — waveform synthesis → STFT → mel. Slower, but produces audio for inspection.

```julia
spec_a, _ = generate_spectrogram(AudioPath(), rng, transcript, scene)
spec_d = generate_spectrogram(DirectPath(), rng, transcript, scene)
report = compare_paths(spec_a, spec_d)
plot_consistency(report)
```
