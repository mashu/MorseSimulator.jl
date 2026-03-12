# CWContestSim.jl

A Julia package for simulating realistic CW Morse amateur radio communications
and generating mel-spectrogram datasets for training Whisper-like audio-to-text
neural networks.

## Overview

The package provides an end-to-end simulation pipeline:

1. **Transcript Generation** — Realistic CW conversations with multiple stations, contest exchanges, ragchewing, DX pileups
2. **Morse Encoding** — Text to timed dots/dashes with operator speed jitter
3. **Signal Synthesis** — CW tone generation with noise, fading, multi-station mixing
4. **Spectrogram Generation** — Mel spectrograms via audio path or fast direct path

## Quick Start

```julia
using CWContestSim, Random

rng = MersenneTwister(42)

# Generate a dataset
config = DatasetConfig(path=DirectPath())
dataset = generate_dataset(rng, 100, config)

# Inspect a sample
sample = dataset[1]
println(sample.label)
display(plot_spectrogram(sample.mel_spectrogram))
```
