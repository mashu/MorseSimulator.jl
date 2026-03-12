"""
    CWContestSim/src/transcript/operators.jl

Operator style archetypes and state machine for CW operator behavior.
Each style controls speed, verbosity, patience, error rates, and
phrase selection through multiple dispatch.
"""

using Random, Distributions

# ============================================================================
# Concrete Operator Styles
# ============================================================================

"""
    FastContestOp <: AbstractOperatorStyle

Fast, efficient contest operator. Short exchanges, high speed,
low verbosity, rarely repeats unless necessary.
"""
struct FastContestOp <: AbstractOperatorStyle end

"""
    CasualRagchewer <: AbstractOperatorStyle

Friendly operator who enjoys longer exchanges with greetings,
names, QTH, weather, rig info. Medium speed, high verbosity.
"""
struct CasualRagchewer <: AbstractOperatorStyle end

"""
    BeginnerOp <: AbstractOperatorStyle

New operator. Slow speed, frequent pauses, occasional errors,
tends to be verbose out of uncertainty.
"""
struct BeginnerOp <: AbstractOperatorStyle end

"""
    DXPileupManager <: AbstractOperatorStyle

DX station running a pileup. Very short transmissions,
sends callsign fragments, manages queue efficiently.
"""
struct DXPileupManager <: AbstractOperatorStyle end

"""
    MidSkillContestOp <: AbstractOperatorStyle

Intermediate contester. Moderate speed, mostly standard exchanges,
occasional verbosity.
"""
struct MidSkillContestOp <: AbstractOperatorStyle end

"""
    QRPOperator <: AbstractOperatorStyle

Low-power operator. May repeat more, patient, expects AGN requests.
"""
struct QRPOperator <: AbstractOperatorStyle end

# ============================================================================
# Style Parameters via Dispatch
# ============================================================================

"""
    mean_wpm(style) -> Float64

Mean CW sending speed in words per minute.
"""
mean_wpm(::FastContestOp)      = 32.0
mean_wpm(::CasualRagchewer)    = 20.0
mean_wpm(::BeginnerOp)         = 12.0
mean_wpm(::DXPileupManager)    = 28.0
mean_wpm(::MidSkillContestOp)  = 24.0
mean_wpm(::QRPOperator)        = 18.0

"""
    wpm_sigma(style) -> Float64

Standard deviation of WPM within a single transmission.
"""
wpm_sigma(::FastContestOp)      = 1.5
wpm_sigma(::CasualRagchewer)    = 2.0
wpm_sigma(::BeginnerOp)         = 3.0
wpm_sigma(::DXPileupManager)    = 1.0
wpm_sigma(::MidSkillContestOp)  = 2.0
wpm_sigma(::QRPOperator)        = 2.5

"""
    verbosity(style) -> Float64

Probability of adding optional phrases (0.0 = minimal, 1.0 = very verbose).
"""
verbosity(::FastContestOp)      = 0.05
verbosity(::CasualRagchewer)    = 0.85
verbosity(::BeginnerOp)         = 0.60
verbosity(::DXPileupManager)    = 0.02
verbosity(::MidSkillContestOp)  = 0.20
verbosity(::QRPOperator)        = 0.30

"""
    patience(style) -> Int

Maximum number of retries before abandoning a QSO.
"""
patience(::FastContestOp)      = 2
patience(::CasualRagchewer)    = 5
patience(::BeginnerOp)         = 4
patience(::DXPileupManager)    = 1
patience(::MidSkillContestOp)  = 3
patience(::QRPOperator)        = 6

"""
    error_rate(style) -> Float64

Probability of a sending error per word.
"""
error_rate(::FastContestOp)      = 0.01
error_rate(::CasualRagchewer)    = 0.02
error_rate(::BeginnerOp)         = 0.08
error_rate(::DXPileupManager)    = 0.005
error_rate(::MidSkillContestOp)  = 0.03
error_rate(::QRPOperator)        = 0.02

"""
    aggressiveness(style) -> Float64

Probability of responding to a CQ in a pileup situation (0-1).
"""
aggressiveness(::FastContestOp)      = 0.90
aggressiveness(::CasualRagchewer)    = 0.30
aggressiveness(::BeginnerOp)         = 0.15
aggressiveness(::DXPileupManager)    = 0.0  # DX station doesn't call
aggressiveness(::MidSkillContestOp)  = 0.70
aggressiveness(::QRPOperator)        = 0.50

"""
    pause_distribution(style) -> Distribution

Distribution of inter-word pause multipliers (1.0 = standard).
"""
pause_distribution(::FastContestOp)      = LogNormal(0.0, 0.1)
pause_distribution(::CasualRagchewer)    = LogNormal(0.2, 0.3)
pause_distribution(::BeginnerOp)         = LogNormal(0.4, 0.5)
pause_distribution(::DXPileupManager)    = LogNormal(0.0, 0.05)
pause_distribution(::MidSkillContestOp)  = LogNormal(0.1, 0.2)
pause_distribution(::QRPOperator)        = LogNormal(0.2, 0.3)

"""
    uses_de(style) -> Float64

Probability that operator includes "DE" between callsigns.
"""
uses_de(::FastContestOp)      = 0.3
uses_de(::CasualRagchewer)    = 0.95
uses_de(::BeginnerOp)         = 0.90
uses_de(::DXPileupManager)    = 0.1
uses_de(::MidSkillContestOp)  = 0.6
uses_de(::QRPOperator)        = 0.80

"""
    uses_tu(style) -> Float64

Probability that operator includes "TU" (thank you) in exchange.
"""
uses_tu(::FastContestOp)      = 0.7
uses_tu(::CasualRagchewer)    = 0.95
uses_tu(::BeginnerOp)         = 0.80
uses_tu(::DXPileupManager)    = 0.5
uses_tu(::MidSkillContestOp)  = 0.6
uses_tu(::QRPOperator)        = 0.85

# ============================================================================
# Operator State Machine
# ============================================================================

struct IdleState      <: AbstractOperatorState end
struct CallingCQ      <: AbstractOperatorState end
struct Listening      <: AbstractOperatorState end
struct Responding     <: AbstractOperatorState end
struct SendingExchange <: AbstractOperatorState end
struct WaitingReply   <: AbstractOperatorState end
struct Retrying       <: AbstractOperatorState end
struct EndingQSO      <: AbstractOperatorState end

# ============================================================================
# Events
# ============================================================================

struct CQHeard        <: AbstractEvent
    from_callsign::String
    signal_strength::Float64
end

struct ResponseHeard  <: AbstractEvent
    from_callsign::String
    signal_strength::Float64
end

struct ExchangeReceived <: AbstractEvent
    from_callsign::String
    exchange_text::String
end

struct ConfirmReceived <: AbstractEvent
    from_callsign::String
end

struct TimeoutExpired <: AbstractEvent end
struct NoResponse     <: AbstractEvent end

# ============================================================================
# Random Style Selection
# ============================================================================

const ALL_STYLES = AbstractOperatorStyle[
    FastContestOp(), CasualRagchewer(), BeginnerOp(),
    DXPileupManager(), MidSkillContestOp(), QRPOperator()
]

const CONTEST_STYLES = AbstractOperatorStyle[
    FastContestOp(), MidSkillContestOp(), QRPOperator(), BeginnerOp()
]

"""
    random_operator_style(rng, mode) -> AbstractOperatorStyle

Select an operator style appropriate for the given conversation mode.
"""
function random_operator_style(rng::AbstractRNG, ::AbstractConversationMode)
    ALL_STYLES[rand(rng, 1:length(ALL_STYLES))]
end
