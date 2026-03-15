# Architecture

## Type hierarchy

All behaviour is controlled through Julia's multiple dispatch on an abstract type hierarchy:

```
AbstractContest          ─── CQWWContest, CQWPXContest, ARRLDXContest, ...
AbstractOperatorStyle    ─── FastContestOp, CasualRagchewer, BeginnerOp, ...
AbstractConversationMode ─── ContestMode, RagchewMode, DXPileupMode, TestMode
AbstractMorseElement     ─── Dot, Dash, SymbolGap, CharGap, WordGap
AbstractEnvelope{T}      ─── RaisedCosineEnvelope{T}, HardEnvelope{T}
AbstractNoiseModel{T}    ─── GaussianNoise{T}, ImpulsiveNoise{T}
AbstractFadingModel{T}   ─── SinusoidalFading{T}, RayleighFading{T}
AbstractSpectrogramPath  ─── AudioPath, DirectPath
```

## Design principles

**Multiple dispatch over conditionals.** Operator behaviour, contest exchanges, and envelope shapes are selected by dispatch rather than `if`/`elseif` chains:

```julia
generate_cq(rng, station, ::FastContestOp)   = "CQ $(station.callsign)"
generate_cq(rng, station, ::CasualRagchewer) = "CQ CQ CQ DE $(station.callsign) K"
```

**Parametric type stability.** Signal-layer types are parameterized on `T<:AbstractFloat` so all arithmetic stays type-stable at compile time.

**Concrete union storage.** The Morse code table stores `Vector{DotOrDash}` (a two-type union of `Dot` and `Dash`) and `TimedMorseEvent` stores `MorseElement` (a five-type union), both of which Julia's small-union optimization handles efficiently.

**Explicit RNG threading.** Every random operation takes an `AbstractRNG` as its first argument for full reproducibility.
