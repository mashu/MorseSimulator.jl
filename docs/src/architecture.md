# Architecture

## Pipeline Layers

```
Conversation → Text → Morse Timing → Signal → Spectrogram
```

## Type Hierarchy

All behavior is controlled through Julia's multiple dispatch system.
The abstract type hierarchy forms the backbone of the package:

```
AbstractContest          ─┬─ CQWWContest, CQWPXContest, ARRLDXContest
                          ├─ IARUHFContest, SPDXContest, WAEContest
                          └─ AllAsianDXContest, GenericSerialContest

AbstractOperatorStyle    ─┬─ FastContestOp, CasualRagchewer
                          ├─ BeginnerOp, DXPileupManager
                          └─ MidSkillContestOp, QRPOperator

AbstractConversationMode ─── ContestMode, RagchewMode, DXPileupMode, TestMode

AbstractMorseElement     ─── Dot, Dash, SymbolGap, CharGap, WordGap

AbstractEnvelope{T}      ─── RaisedCosineEnvelope{T}, HardEnvelope{T}

AbstractNoiseModel{T}    ─── GaussianNoise{T}, ImpulsiveNoise{T}

AbstractFadingModel{T}   ─── SinusoidalFading{T}, RayleighFading{T}

AbstractSpectrogramPath  ─── AudioPath, DirectPath
```

## Design Principles

### Multiple Dispatch over Conditionals

Instead of:
```julia
# Bad: type checking
if style isa FastContestOp
    return "CQ $(station.callsign)"
elseif style isa CasualRagchewer
    ...
end
```

We use dispatch:
```julia
# Good: multiple dispatch
generate_cq(rng, station, ::FastContestOp) = "CQ $(station.callsign)"
generate_cq(rng, station, ::CasualRagchewer) = "CQ CQ CQ DE ..."
```

### Parametric Type Stability

All signal-layer types are parameterized on `T<:AbstractFloat` for
compile-time type inference and zero-cost abstraction.

### Explicit RNG Threading

All random operations accept an `AbstractRNG` as first argument
for full reproducibility.
