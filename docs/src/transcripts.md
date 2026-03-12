# Phase 1: Transcript Simulation

## Overview

Transcript generation simulates realistic CW amateur radio conversations
involving up to 6 stations. The system models:

- **Contest exchanges** with proper format and serial numbers
- **Ragchew conversations** with greetings, names, QTH, rig info
- **DX pileups** with simultaneous callers and station selection
- **Test transmissions** with test patterns

## Band Scene

A `BandScene` encapsulates the radio environment:

```julia
scene = BandScene(rng;
    num_stations = 4,
    contest = CQWWContest(),
    mode = ContestMode(),
    propagation = moderate_propagation(),
    noise_floor_db = -20.0
)
```

## Station Model

Each `Station` has an operator style that controls behavior via dispatch:

```julia
station = Station(rng, "W1ABC", FastContestOp())
```

## Conversation Modes

```julia
# Contest: fast exchanges with CQ runs
transcript = generate_transcript(rng; mode=ContestMode())

# Ragchew: extended QSO with personal info
transcript = generate_transcript(rng; mode=RagchewMode())

# DX Pileup: multiple callers, DX station picks one
transcript = generate_transcript(rng; mode=DXPileupMode())

# Test: VVV patterns, QRL checks
transcript = generate_transcript(rng; mode=TestMode())
```

## Realistic Features

- Sending errors with corrections (`RST 579 EEE 589`)
- Partial callsign requests (`ABC?`, `?7NA`)
- Retry logic (`AGN`, `PSE AGN`, `QRS PSE`)
- Variable verbosity per operator style
- Overlapping transmissions (collisions)

## Output Format

```
<START> CQ CQ DE SO5KM SO5KM DE SO7NA UR RST 5NN TU SO7NA DE SO5KM 5NN TU K <END>
```
