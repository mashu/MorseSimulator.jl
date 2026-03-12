"""
    CWContestSim/src/types.jl

Core abstract type hierarchy for the CW Contest Simulator.
All concrete types in the package derive from these abstract types,
enabling clean multiple dispatch throughout the system.
"""

# ============================================================================
# Phase 1: Transcript Layer Types
# ============================================================================

"""
    AbstractContest

Root type for all contest/operating mode definitions.
Subtypes define exchange rules, required fields, and scoring.
"""
abstract type AbstractContest end

"""
    AbstractOperatorStyle

Behavioral archetype for CW operators.
Subtypes control speed, verbosity, patience, and phrase selection.
"""
abstract type AbstractOperatorStyle end

"""
    AbstractOperatorState

State in the operator finite state machine.
Transitions between states are governed by dispatch on
(current_state, event) pairs.
"""
abstract type AbstractOperatorState end

"""
    AbstractEvent

Events that drive state transitions in the simulation.
Events may originate from timers, received transmissions,
or stochastic operator decisions.
"""
abstract type AbstractEvent end

"""
    AbstractConversationMode

Top-level operating mode (contest, ragchew, pileup, test).
Determines which conversation generator is dispatched.
"""
abstract type AbstractConversationMode end

# ============================================================================
# Phase 2: Signal Layer Types
# ============================================================================

"""
    AbstractMorseElement

Atomic unit of Morse timing: dots, dashes, and gaps.
Parameterized by duration type for type stability.
"""
abstract type AbstractMorseElement end

"""
    AbstractEnvelope{T<:AbstractFloat}

Amplitude envelope applied to CW tone signals.
`T` is the sample precision (Float32 or Float64).
"""
abstract type AbstractEnvelope{T<:AbstractFloat} end

"""
    AbstractChannelEffect{T<:AbstractFloat}

A propagation or channel effect applied to signals.
Subtypes include noise, fading (QSB), frequency offset, etc.
"""
abstract type AbstractChannelEffect{T<:AbstractFloat} end

"""
    AbstractNoiseModel{T<:AbstractFloat}

Noise generation model for the channel.
"""
abstract type AbstractNoiseModel{T<:AbstractFloat} end

"""
    AbstractFadingModel{T<:AbstractFloat}

Signal fading (QSB) model.
"""
abstract type AbstractFadingModel{T<:AbstractFloat} end

# ============================================================================
# Spectrogram Layer Types
# ============================================================================

"""
    AbstractSpectrogramPath

Strategy for mel-spectrogram generation.
Two concrete paths: AudioPath (via waveform) and DirectPath (analytic).
"""
abstract type AbstractSpectrogramPath end

"""
    AbstractConsistencyMetric

Metric for comparing spectrograms from different generation paths.
"""
abstract type AbstractConsistencyMetric end
