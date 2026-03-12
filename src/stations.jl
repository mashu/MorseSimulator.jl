"""
    CWContestSim/src/transcript/stations.jl

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

"""
    Station(rng, callsign, style) -> Station

Construct a station with random signal characteristics.
"""
function Station(rng::AbstractRNG, callsign::String, style::S) where {S<:AbstractOperatorStyle}
    tone = 400.0 + rand(rng) * 500.0  # 400-900 Hz
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
