"""
    MorseSimulator.jl/src/transcript/stations.jl

Station model representing an individual amateur radio station
with callsign, operator style, signal characteristics, and
exchange state.
"""

using Random, Distributions

"""
    Station{S<:AbstractOperatorStyle}

A simulated amateur radio station parameterized by operator style.

# Fields
- `callsign::String` — station callsign
- `style::S` — operator behavioral style
- `tone_freq::Float64` — preferred CW tone frequency (Hz)
- `signal_amplitude::Float64` — relative signal strength (linear)
- `serial_counter::Int` — running serial number for contests
- `state::AbstractOperatorState` — current FSM state
- `retry_count::Int` — current retry counter
- `qso_partner::String` — current QSO partner callsign
- `name::String` — operator name (for ragchew)
- `qth::String` — station location (for ragchew)
"""
mutable struct Station{S<:AbstractOperatorStyle}
    callsign::String
    style::S
    tone_freq::Float64
    signal_amplitude::Float64
    serial_counter::Int
    state::AbstractOperatorState
    retry_count::Int
    qso_partner::String
    name::String
    qth::String
end

# Responders: 0–350 Hz from caller; same-tone possible; offset distribution biases toward smaller.
const TONE_OFFSET_MAX = 350.0             # max offset magnitude (Hz) from caller
const TONE_OFFSET_MIN = 10.0              # min nonzero offset so stations stay distinguishable
const TONE_BAND = (400.0, 900.0)          # Hz (full band for mel; callers sit in 400–700)
const TONE_SAME_AS_CALLER_PROB = 0.15     # probability a responder uses caller's tone exactly
# Offset magnitude bias: use power law so smaller offsets (closer to caller) more likely
const TONE_OFFSET_POWER = 1.5             # magnitude = MIN + (MAX-MIN) * rand()^POWER

"""
    Station(rng, callsign, style; ref_freq=nothing, is_caller=false) -> Station

Construct a station with random signal characteristics.

If `ref_freq` is set, the station's tone is relative to the caller:
- Caller (`is_caller=true`): uses `ref_freq` exactly.
- Others: with probability `TONE_SAME_AS_CALLER_PROB` use same tone; otherwise
  `ref_freq` ± offset with magnitude in [TONE_OFFSET_MIN, TONE_OFFSET_MAX] (10–350 Hz),
  with a probabilistic distribution that favors smaller offsets. Clamped to `TONE_BAND`.
If `ref_freq` is not set, tone is uniform in 400–900 Hz (legacy behavior).
"""
function Station(rng::AbstractRNG, callsign::String, style::S;
                 ref_freq::Union{Float64,Nothing} = nothing,
                 is_caller::Bool = false) where {S<:AbstractOperatorStyle}
    tone = if ref_freq !== nothing
        if is_caller
            Float64(ref_freq)
        else
            if rand(rng) < TONE_SAME_AS_CALLER_PROB
                Float64(ref_freq)
            else
                # 0–350 Hz from caller; bias toward smaller offset (rand^power)
                span = TONE_OFFSET_MAX - TONE_OFFSET_MIN
                magnitude = TONE_OFFSET_MIN + span * Float64(rand(rng))^TONE_OFFSET_POWER
                offset = magnitude * (rand(rng, Bool) ? 1.0 : -1.0)
                clamp(ref_freq + offset, TONE_BAND[1], TONE_BAND[2])
            end
        end
    else
        400.0 + rand(rng) * (TONE_BAND[2] - TONE_BAND[1])
    end
    amp = exp(randn(rng) * 0.5)       # LogNormal amplitude
    name = random_op_name(rng)
    qth = random_qth(rng)
    Station{S}(callsign, style, tone, amp, 1, IdleState(), 0, "", name, qth)
end

"""
    advance_serial!(station) -> Int

Increment and return the serial number.
"""
function advance_serial!(station::Station)
    station.serial_counter += 1
    return station.serial_counter - 1
end

"""
    reset_qso!(station)

Reset station state for a new QSO.
"""
function reset_qso!(station::Station)
    station.state = IdleState()
    station.retry_count = 0
    station.qso_partner = ""
    nothing
end

"""
    instantaneous_wpm(rng, station) -> Float64

Sample the current sending speed for a station, based on
its style's mean and sigma.
"""
function instantaneous_wpm(rng::AbstractRNG, station::Station)
    μ = mean_wpm(station.style)
    σ = wpm_sigma(station.style)
    max(8.0, μ + randn(rng) * σ)
end

# ============================================================================
# Random Operator Names and QTH
# ============================================================================

const OP_NAMES = [
    "JOHN", "MIKE", "ALEX", "BOB", "TOM", "JAN", "PIOTR", "HANS",
    "YURI", "PIERRE", "CARLOS", "TARO", "SVEN", "LARS", "FRITZ",
    "IVAN", "DMITRI", "PAOLO", "MARCO", "ANDRZEJ", "WOJTEK", "ADAM",
    "DAVID", "JAMES", "PETER", "FRANK", "KURT", "OTTO", "KENJI",
    "LEE", "CHEN", "RAVI", "AHMED", "ALI", "JOSE", "MARIA"
]

const QTH_CITIES = [
    "WARSAW", "BERLIN", "LONDON", "PARIS", "ROME", "MADRID",
    "PRAGUE", "MOSCOW", "TOKYO", "NEW YORK", "CHICAGO", "SAO PAULO",
    "SYDNEY", "STOCKHOLM", "OSLO", "HELSINKI", "AMSTERDAM", "VIENNA",
    "BRUSSELS", "BUDAPEST", "BUCHAREST", "SOFIA", "ZAGREB", "KIEV",
    "SEOUL", "BEIJING", "MUMBAI", "DELHI", "BANGKOK", "MANILA",
    "KRAKOW", "GDANSK", "WROCLAW", "POZNAN", "LODZ", "KATOWICE"
]

random_op_name(rng::AbstractRNG) = OP_NAMES[rand(rng, 1:length(OP_NAMES))]
random_qth(rng::AbstractRNG) = QTH_CITIES[rand(rng, 1:length(QTH_CITIES))]
