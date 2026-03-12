"""
    CWContestSim/src/transcript/contests.jl

Contest definitions with exchange rules, required fields, and
phrase generation logic. Each contest type uses dispatch to
produce realistic exchanges.
"""

using Random, Distributions

# ============================================================================
# Concrete Contest Types
# ============================================================================

"""
    CQWWContest <: AbstractContest

CQ World Wide DX Contest. Exchange: RST + CQ Zone (1-40).
"""
struct CQWWContest <: AbstractContest end

"""
    CQWPXContest <: AbstractContest

CQ World Wide WPX Contest. Exchange: RST + serial number.
"""
struct CQWPXContest <: AbstractContest end

"""
    ARRLDXContest <: AbstractContest

ARRL DX Contest. US/VE sends RST + state/province.
DX sends RST + power.
"""
struct ARRLDXContest <: AbstractContest end

"""
    IARUHFContest <: AbstractContest

IARU HF Championship. Exchange: RST + ITU zone or HQ designator.
"""
struct IARUHFContest <: AbstractContest end

"""
    SPDXContest <: AbstractContest

SP DX Contest. Exchange: RST + serial (non-SP) or RST + province (SP).
"""
struct SPDXContest <: AbstractContest end

"""
    WAEContest <: AbstractContest

Worked All Europe DX Contest. Exchange: RST + serial. Includes QTC mechanism.
"""
struct WAEContest <: AbstractContest end

"""
    AllAsianDXContest <: AbstractContest

All Asian DX Contest. Exchange: RST + operator age.
"""
struct AllAsianDXContest <: AbstractContest end

"""
    GenericSerialContest <: AbstractContest

Generic contest with RST + serial number exchange.
Covers many club and national contests.
"""
struct GenericSerialContest <: AbstractContest end

# ============================================================================
# Exchange Data
# ============================================================================

"""
    ExchangeData

Holds the components of a contest exchange for one station.
Fields are optional (empty string if not applicable).
"""
struct ExchangeData
    rst::String
    serial::String
    zone::String
    state::String
    power::String
    age::String
    extra::String
end

ExchangeData(; rst="5NN", serial="", zone="", state="", power="", age="", extra="") =
    ExchangeData(rst, serial, zone, state, power, age, extra)

# ============================================================================
# Exchange Generation via Dispatch
# ============================================================================

const CQ_ZONES = string.(1:40)
const US_STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN",
    "MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
const SP_PROVINCES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]

"""
    generate_exchange(contest, rng, serial_num) -> ExchangeData

Generate a contest-specific exchange. Dispatches on contest type.
"""
function generate_exchange(::CQWWContest, rng::AbstractRNG, serial_num::Int)
    zone = CQ_ZONES[rand(rng, 1:length(CQ_ZONES))]
    ExchangeData(rst="5NN", zone=zone)
end

function generate_exchange(::CQWPXContest, rng::AbstractRNG, serial_num::Int)
    ExchangeData(rst="5NN", serial=lpad(serial_num, 3, '0'))
end

function generate_exchange(::ARRLDXContest, rng::AbstractRNG, serial_num::Int)
    # Randomly either US station (state) or DX station (power)
    if rand(rng, Bool)
        state = US_STATES[rand(rng, 1:length(US_STATES))]
        ExchangeData(rst="5NN", state=state)
    else
        pwr = rand(rng, [100, 200, 400, 500, 1000])
        ExchangeData(rst="5NN", power=string(pwr))
    end
end

function generate_exchange(::IARUHFContest, rng::AbstractRNG, serial_num::Int)
    zone = string(rand(rng, 1:90))
    ExchangeData(rst="5NN", zone=zone)
end

function generate_exchange(::SPDXContest, rng::AbstractRNG, serial_num::Int)
    if rand(rng, Bool)
        prov = SP_PROVINCES[rand(rng, 1:length(SP_PROVINCES))]
        ExchangeData(rst="5NN", extra=prov)
    else
        ExchangeData(rst="5NN", serial=lpad(serial_num, 3, '0'))
    end
end

function generate_exchange(::WAEContest, rng::AbstractRNG, serial_num::Int)
    ExchangeData(rst="5NN", serial=lpad(serial_num, 3, '0'))
end

function generate_exchange(::AllAsianDXContest, rng::AbstractRNG, serial_num::Int)
    age = string(rand(rng, 18:85))
    ExchangeData(rst="5NN", age=age)
end

function generate_exchange(::GenericSerialContest, rng::AbstractRNG, serial_num::Int)
    ExchangeData(rst="5NN", serial=lpad(serial_num, 3, '0'))
end

# ============================================================================
# Format Exchange to Text
# ============================================================================

"""
    format_exchange(contest, exch) -> String

Format an ExchangeData into the text string sent over CW.
"""
format_exchange(::CQWWContest, exch::ExchangeData) =
    "$(exch.rst) $(exch.zone)"

format_exchange(::CQWPXContest, exch::ExchangeData) =
    "$(exch.rst) $(exch.serial)"

function format_exchange(::ARRLDXContest, exch::ExchangeData)
    isempty(exch.state) || return "$(exch.rst) $(exch.state)"
    return "$(exch.rst) $(exch.power)"
end

format_exchange(::IARUHFContest, exch::ExchangeData) =
    "$(exch.rst) $(exch.zone)"

function format_exchange(::SPDXContest, exch::ExchangeData)
    isempty(exch.extra) || return "$(exch.rst) $(exch.extra)"
    return "$(exch.rst) $(exch.serial)"
end

format_exchange(::WAEContest, exch::ExchangeData) =
    "$(exch.rst) $(exch.serial)"

format_exchange(::AllAsianDXContest, exch::ExchangeData) =
    "$(exch.rst) $(exch.age)"

format_exchange(::GenericSerialContest, exch::ExchangeData) =
    "$(exch.rst) $(exch.serial)"

# ============================================================================
# Contest Name for Display
# ============================================================================

contest_name(::CQWWContest) = "CQ WW DX"
contest_name(::CQWPXContest) = "CQ WPX"
contest_name(::ARRLDXContest) = "ARRL DX"
contest_name(::IARUHFContest) = "IARU HF"
contest_name(::SPDXContest) = "SP DX"
contest_name(::WAEContest) = "WAE"
contest_name(::AllAsianDXContest) = "All Asian DX"
contest_name(::GenericSerialContest) = "Generic Serial"

# ============================================================================
# Random Contest Selection
# ============================================================================

const ALL_CONTESTS = [
    CQWWContest(), CQWPXContest(), ARRLDXContest(),
    IARUHFContest(), SPDXContest(), WAEContest(),
    AllAsianDXContest(), GenericSerialContest()
]

"""
    random_contest(rng) -> AbstractContest

Select a random contest type.
"""
random_contest(rng::AbstractRNG) = ALL_CONTESTS[rand(rng, 1:length(ALL_CONTESTS))]
