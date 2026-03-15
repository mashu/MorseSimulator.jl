# Transcripts

## Conversation modes

The simulator supports four conversation modes, each producing a different transcript structure:

| Mode | Description |
|------|-------------|
| `ContestMode()` | Rapid exchange of signal reports, serial numbers, and contest-specific data |
| `RagchewMode()` | Extended casual conversation between two operators |
| `DXPileupMode()` | One DX station working a pile of callers |
| `TestMode()` | Simple CQ and response pairs for debugging |

## Operator styles

Each station is assigned an operator style that controls speed, jitter, verbosity, and error rate:

| Style | Typical WPM | Character |
|-------|-------------|-----------|
| `FastContestOp` | 35–40 | Terse, fast, contest-optimized |
| `MidSkillContestOp` | 25–30 | Competent contester |
| `CasualRagchewer` | 18–22 | Verbose, uses "DE", long CQs |
| `BeginnerOp` | 12–18 | Slow, makes errors, retries |
| `DXPileupManager` | 28–35 | Short, directive transmissions |
| `QRPOperator` | 15–20 | Low power, careful sending |

## Contest formats

Eight contest types are supported, each defining exchange rules, required fields, and formatting:

`CQWWContest`, `CQWPXContest`, `ARRLDXContest`, `IARUHFContest`, `SPDXContest`, `WAEContest`, `AllAsianDXContest`, `GenericSerialContest`.

## Band scene

A `BandScene` models the RF environment: multiple stations with different frequencies, amplitudes, and propagation conditions sharing a band segment. The transcript records each station's transmissions with time offsets so overlapping signals are possible.

## Training labels

Labels use a structured flat-text format with special tokens:

```
<START> [TS] [S1] CQ CQ W1ABC [TE] [TS] [S2] W1ABC DE N2XYZ 5NN 03 [TE] <END>
```

`[TS]`/`[TE]` mark transmission boundaries; `[S1]`–`[S6]` identify the station.
