# MorseSimulator.jl

[![CI](https://github.com/mashu/MorseSimulator.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/MorseSimulator.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mashu/MorseSimulator.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/MorseSimulator.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://mashu.github.io/MorseSimulator.jl/stable/)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mashu.github.io/MorseSimulator.jl/dev/)

Simulate realistic CW Morse amateur radio communications and generate mel-spectrogram datasets for training neural network decoders.

## Pipeline

```
Conversation → Morse Timing → Signal → Mel Spectrogram
```

Multiple contest formats, operator styles, propagation effects, and noise models are combined to produce diverse training data via Julia's multiple dispatch.

## Quick start

```julia
using MorseSimulator, Random

rng = MersenneTwister(42)
config = DatasetConfig(path=DirectPath())

# One sample (mel spectrogram + training label)
sample = generate_sample(rng, config)

# With audio for inspection
sample, audio = generate_sample_with_audio(rng, config)
save_audio("test.wav", audio)

# Batch
dataset = generate_dataset(rng, 100, config)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mashu/MorseSimulator.jl")
```

## Documentation

See the [full documentation](https://mashu.github.io/MorseSimulator.jl/dev/) for architecture details, API reference, and examples.
