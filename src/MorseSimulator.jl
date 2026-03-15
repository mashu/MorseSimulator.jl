"""
    MorseSimulator

A Julia package for simulating realistic CW Morse amateur radio
communications and generating mel-spectrogram datasets for training
Whisper-like audio-to-text neural networks.

## Architecture

The simulation consists of five conceptual layers:

    Conversation → Text → Morse Timing → Signal → Spectrogram

## Quick Start

```julia
using MorseSimulator, Random

rng = MersenneTwister(42)
config = DatasetConfig(path=DirectPath())

# Option A: Generate sample (spectrogram + optional audio) — new transcript each time
sample = generate_sample(rng, config)
sample, audio = generate_sample_with_audio(rng, config)
save_audio("test.wav", audio)
# Use a varying rng (e.g. different seed or next rand) to get different content.

# Option B: Generate transcript first, inspect, then generate from it
scene = BandScene(rng; num_stations=3)
transcript = generate_transcript(rng, scene)
plot_transcript(transcript)
# Now generate spectrogram or audio from this exact transcript
sample = generate_sample(rng, config, transcript, scene)
sample, audio = generate_sample_with_audio(rng, config, transcript, scene)
save_audio("inspection.wav", audio)

# Compare generation paths (same transcript → audio path vs direct path)
spec_audio, _ = generate_spectrogram(AudioPath(), rng, transcript, scene)
spec_direct = generate_spectrogram(DirectPath(), rng, transcript, scene)
report = compare_paths(spec_audio, spec_direct)
plot_consistency(report)
```
"""
module MorseSimulator

using Random
using Distributions
using FFTW
using DSP
using LibSndFile
using UnicodePlots

# ============================================================================
# Core Types (loaded first — all other files depend on these)
# ============================================================================
include("types.jl")

# ============================================================================
# Phase 1: Transcript Layer
# ============================================================================
include("transcript/callsigns.jl")
include("transcript/contests.jl")
include("transcript/operators.jl")
include("transcript/stations.jl")
include("transcript/phrases.jl")
include("transcript/transmission.jl")
include("transcript/bandscene.jl")
include("transcript/conversation.jl")

# ============================================================================
# Phase 2: Morse Timing Layer
# ============================================================================
include("morse/code.jl")
include("morse/timing.jl")
include("morse/encoder.jl")
include("alignment.jl")

# ============================================================================
# Phase 2: Signal Layer
# ============================================================================
include("signal/envelope.jl")
include("signal/tone.jl")
include("signal/channel.jl")
include("signal/mixer.jl")
include("signal/audio.jl")

# ============================================================================
# Phase 2: Spectrogram Layer
# ============================================================================
include("spectrogram/mel.jl")
include("spectrogram/stft.jl")
include("spectrogram/audio_path.jl")
include("spectrogram/direct_path.jl")
include("spectrogram/consistency.jl")

# ============================================================================
# Visualization
# ============================================================================
include("visualization/plotting.jl")

# ============================================================================
# High-Level Dataset API
# ============================================================================
include("dataset.jl")

# ============================================================================
# Exports
# ============================================================================

# Constants
export ALL_CONTESTS, ALL_STYLES

# Types — Abstract
export AbstractContest, AbstractOperatorStyle, AbstractOperatorState
export AbstractEvent, AbstractConversationMode
export AbstractMorseElement, AbstractEnvelope, AbstractChannelEffect
export AbstractNoiseModel, AbstractFadingModel
export AbstractSpectrogramPath, AbstractConsistencyMetric

# Types — Contests
export CQWWContest, CQWPXContest, ARRLDXContest, IARUHFContest
export SPDXContest, WAEContest, AllAsianDXContest, GenericSerialContest
export ExchangeData

# Types — Operators
export FastContestOp, CasualRagchewer, BeginnerOp, DXPileupManager
export MidSkillContestOp, QRPOperator
export IdleState, CallingCQ, Listening, Responding
export SendingExchange, WaitingReply, Retrying, EndingQSO

# Types — Conversation Modes
export ContestMode, RagchewMode, DXPileupMode, TestMode

# Types — Station and Transcript
export Station, Transmission, Transcript, BandScene
export PropagationCondition

# Types — Morse
export Dot, Dash, SymbolGap, CharGap, WordGap
export TimingParams, TimedMorseEvent
export StationMorseEvents, SceneMorseEvents

# Types — Signal
export RaisedCosineEnvelope, HardEnvelope
export GaussianNoise, ImpulsiveNoise
export SinusoidalFading, RayleighFading
export ChannelConfig, MixedSignal, AudioConfig

# Types — Spectrogram
export MelFilterbank, STFTConfig
export AudioPath, DirectPath
export SpectrogramResult
export L2SpectralError, CosineSimilarity, KLDivergence, MeanAbsoluteError
export ConsistencyReport

# Types — Dataset & Alignment
export DatasetSample, DatasetConfig
export AbstractTokenTiming, TokenTiming, NoTiming

# Functions — Callsigns
export generate_callsign, generate_callsigns

# Functions — Contests
export generate_exchange, format_exchange, contest_name, random_contest

# Functions — Operators
export mean_wpm, wpm_sigma, verbosity, patience, error_rate
export aggressiveness, pause_distribution, uses_de, uses_tu

# Functions — Stations
export advance_serial!, reset_qso!, instantaneous_wpm

# Functions — Phrases
export generate_cq, generate_response, generate_exchange_phrase
export generate_confirmation, generate_ragchew_extras
export maybe_insert_error, generate_retry, random_mode

# Functions — Transcripts
export generate_transcript, generate_transcripts
export flat_text, annotated_text, estimate_duration

# Functions — Morse
export char_to_morse, is_prosign, prosign_to_morse
export dot_units, is_keyed
export text_to_timed_events, total_duration
export encode_transcript

# Functions — Signal
export generate_envelope!, generate_envelope
export generate_tone!, generate_cw_signal, generate_station_signal
export apply_noise!, apply_fading!, apply_channel!
export mix_signals
export save_audio, save_wav, load_wav

# Functions — Spectrogram
export hz_to_mel, mel_to_hz
export apply_filterbank
export compute_stft, power_spectrogram, log_power_spectrogram
export generate_spectrogram
export compare, compare_paths

# Functions — Propagation
export good_propagation, moderate_propagation, poor_propagation
export signal_strength

# Functions — Visualization
export plot_transcript, plot_spectrogram, plot_waveform
export plot_morse_timing, plot_consistency

# Functions — Dataset
export generate_sample, generate_dataset
export generate_sample_with_audio

end # module
