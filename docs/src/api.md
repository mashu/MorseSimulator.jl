# API Reference

## Dataset Generation

```@docs
DatasetConfig
DatasetSample
generate_sample
generate_dataset
generate_sample_with_audio
```

## Transcript Generation

```@docs
generate_transcript
generate_transcripts
BandScene
Station
Transcript
Transmission
flat_text
annotated_text
```

## Contests

```@docs
CQWWContest
CQWPXContest
ARRLDXContest
IARUHFContest
SPDXContest
WAEContest
AllAsianDXContest
GenericSerialContest
ExchangeData
generate_exchange
format_exchange
```

## Operator Styles

```@docs
FastContestOp
CasualRagchewer
BeginnerOp
DXPileupManager
MidSkillContestOp
QRPOperator
mean_wpm
verbosity
patience
error_rate
aggressiveness
```

## Conversation Modes

```@docs
ContestMode
RagchewMode
DXPileupMode
TestMode
```

## Morse Code

```@docs
Dot
Dash
SymbolGap
CharGap
WordGap
char_to_morse
is_prosign
dot_units
is_keyed
TimingParams
TimedMorseEvent
text_to_timed_events
```

## Signal Generation

```@docs
RaisedCosineEnvelope
HardEnvelope
generate_envelope
generate_cw_signal
generate_station_signal
GaussianNoise
ImpulsiveNoise
SinusoidalFading
RayleighFading
ChannelConfig
MixedSignal
mix_signals
save_audio
```

## Spectrogram

```@docs
MelFilterbank
hz_to_mel
mel_to_hz
apply_filterbank
STFTConfig
compute_stft
power_spectrogram
AudioPath
DirectPath
SpectrogramResult
generate_spectrogram
```

## Consistency Metrics

```@docs
L2SpectralError
CosineSimilarity
KLDivergence
MeanAbsoluteError
ConsistencyReport
compare
compare_paths
```

## Visualization

```@docs
plot_transcript
plot_spectrogram
plot_waveform
plot_morse_timing
plot_consistency
```
