"""
    MorseSimulator.jl/src/transcript/phrases.jl

Phrase template system for generating realistic CW text.
Templates use dispatch on operator style and conversation mode
to produce diverse, realistic transmissions.
"""

using Random

# ============================================================================
# Conversation Modes
# ============================================================================

"""
    ContestMode <: AbstractConversationMode

Contest-style QSOs: CQ, call, exchange, TU. Short and structured.
"""
struct ContestMode    <: AbstractConversationMode end

"""
    RagchewMode <: AbstractConversationMode

Casual ragchew: longer exchanges, greetings, names, etc.
"""
struct RagchewMode    <: AbstractConversationMode end

"""
    DXPileupMode <: AbstractConversationMode

DX pileup: multiple callers, retries, confirmations.
"""
struct DXPileupMode   <: AbstractConversationMode end

"""
    TestMode <: AbstractConversationMode

Minimal test QSOs for debugging or simple demos.
"""
struct TestMode       <: AbstractConversationMode end

mode_name(::ContestMode)  = "Contest"
mode_name(::RagchewMode)  = "Ragchew"
mode_name(::DXPileupMode) = "DX Pileup"
mode_name(::TestMode)     = "Test"

const ALL_MODES = AbstractConversationMode[
    ContestMode(), RagchewMode(), DXPileupMode(), TestMode()
]
const MODE_WEIGHTS = [0.55, 0.20, 0.15, 0.10]

"""
    random_mode(rng) -> AbstractConversationMode

Select a random conversation mode based on configured probabilities.
"""
function random_mode(rng::AbstractRNG)
    r = rand(rng)
    cumw = 0.0
    for (mode, w) in zip(ALL_MODES, MODE_WEIGHTS)
        cumw += w
        r <= cumw && return mode
    end
    return ALL_MODES[end]
end

# ============================================================================
# CQ Phrases
# ============================================================================

"""
    generate_cq(rng, station, style) -> String

Generate a CQ call for the given station.
"""
function generate_cq(rng::AbstractRNG, station::Station, ::FastContestOp)
    n = rand(rng, 1:2)
    cqs = join(fill("CQ", n), " ")
    return "$cqs $(station.callsign)"
end

function generate_cq(rng::AbstractRNG, station::Station, ::CasualRagchewer)
    n = rand(rng, 2:3)
    cqs = join(fill("CQ", n), " ")
    calls = rand(rng, Bool) ? "$(station.callsign) $(station.callsign)" : station.callsign
    return "$cqs DE $calls K"
end

function generate_cq(rng::AbstractRNG, station::Station, ::BeginnerOp)
    return "CQ CQ CQ DE $(station.callsign) $(station.callsign) $(station.callsign) K"
end

function generate_cq(rng::AbstractRNG, station::Station, ::DXPileupManager)
    prefix = rand(rng, ["CQ DX", "CQ NA", "CQ EU", "CQ AS", "CQ"])
    return "$prefix $(station.callsign)"
end

function generate_cq(rng::AbstractRNG, station::Station, ::MidSkillContestOp)
    n = rand(rng, 1:2)
    cqs = join(fill("CQ", n), " ")
    use_test = rand(rng, ["", " TEST"])
    return "$(cqs)$(use_test) $(station.callsign)"
end

function generate_cq(rng::AbstractRNG, station::Station, ::QRPOperator)
    return "CQ CQ DE $(station.callsign) QRP K"
end

# ============================================================================
# Response Phrases (Calling station responds to CQ)
# ============================================================================

"""
    generate_response(rng, caller, cq_station, style) -> String

Generate a response to a CQ call.
"""
function generate_response(rng::AbstractRNG, caller::Station, cq_call::String, ::FastContestOp)
    return caller.callsign
end

function generate_response(rng::AbstractRNG, caller::Station, cq_call::String, ::CasualRagchewer)
    return "$cq_call DE $(caller.callsign) $(caller.callsign) K"
end

function generate_response(rng::AbstractRNG, caller::Station, cq_call::String, ::BeginnerOp)
    return "$cq_call DE $(caller.callsign) $(caller.callsign) K"
end

function generate_response(rng::AbstractRNG, caller::Station, cq_call::String, ::DXPileupManager)
    return caller.callsign  # In pileup, DX doesn't call; this is for other styles calling DX
end

function generate_response(rng::AbstractRNG, caller::Station, cq_call::String, ::MidSkillContestOp)
    use_de = rand(rng, Bool) ? "DE " : ""
    return "$cq_call $(use_de)$(caller.callsign)"
end

function generate_response(rng::AbstractRNG, caller::Station, cq_call::String, ::QRPOperator)
    return "$cq_call DE $(caller.callsign)"
end

# ============================================================================
# Exchange Phrases (Contest-specific exchange delivery)
# ============================================================================

"""
    generate_exchange_phrase(rng, from_station, to_call, exchange_text, style) -> String

Generate the full exchange transmission text.
"""
function generate_exchange_phrase(rng::AbstractRNG, from::Station, to_call::String,
                                  exch_text::String, ::FastContestOp)
    return "$to_call $exch_text"
end

function generate_exchange_phrase(rng::AbstractRNG, from::Station, to_call::String,
                                  exch_text::String, ::CasualRagchewer)
    de = rand(rng) < uses_de(from.style) ? " DE $(from.callsign)" : ""
    tu = rand(rng) < uses_tu(from.style) ? " TU" : ""
    return "$to_call$(de) UR RST $exch_text$(tu)"
end

function generate_exchange_phrase(rng::AbstractRNG, from::Station, to_call::String,
                                  exch_text::String, ::BeginnerOp)
    return "$to_call DE $(from.callsign) UR RST $exch_text TU K"
end

function generate_exchange_phrase(rng::AbstractRNG, from::Station, to_call::String,
                                  exch_text::String, ::DXPileupManager)
    return "$to_call $exch_text"
end

function generate_exchange_phrase(rng::AbstractRNG, from::Station, to_call::String,
                                  exch_text::String, ::MidSkillContestOp)
    de = rand(rng) < uses_de(from.style) ? " DE $(from.callsign)" : ""
    tu = rand(rng) < uses_tu(from.style) ? " TU" : ""
    return "$to_call$(de) $exch_text$(tu)"
end

function generate_exchange_phrase(rng::AbstractRNG, from::Station, to_call::String,
                                  exch_text::String, ::QRPOperator)
    return "$to_call DE $(from.callsign) UR RST $exch_text TU"
end

# ============================================================================
# Confirmation / End-of-QSO Phrases
# ============================================================================

"""
    generate_confirmation(rng, from_station, to_call, style) -> String

Generate QSO confirmation/ending text.
"""
function generate_confirmation(rng::AbstractRNG, from::Station, to_call::String, ::FastContestOp)
    ending = rand(rng, ["TU", "R TU", "QSL TU", "73"])
    return "$to_call $ending $(from.callsign)"
end

function generate_confirmation(rng::AbstractRNG, from::Station, to_call::String, ::CasualRagchewer)
    extras = rand(rng, ["TNX FER QSO", "HPE CU AGN", "GL DX", "FB OM", "VY 73"])
    return "$to_call DE $(from.callsign) R $extras 73 SK"
end

function generate_confirmation(rng::AbstractRNG, from::Station, to_call::String, ::BeginnerOp)
    return "$to_call DE $(from.callsign) R TNX 73 SK"
end

function generate_confirmation(rng::AbstractRNG, from::Station, to_call::String, ::DXPileupManager)
    return "TU $(from.callsign)"
end

function generate_confirmation(rng::AbstractRNG, from::Station, to_call::String, ::MidSkillContestOp)
    ending = rand(rng, ["TU", "R TU", "QSL", "73 TU"])
    return "$to_call $ending $(from.callsign)"
end

function generate_confirmation(rng::AbstractRNG, from::Station, to_call::String, ::QRPOperator)
    return "$to_call DE $(from.callsign) R TNX FER QSO 73 TU SK"
end

# ============================================================================
# Ragchew Extra Phrases
# ============================================================================

"""
    generate_ragchew_extras(rng, station, style) -> Vector{String}

Generate additional ragchew content (name, QTH, rig, wx, etc.).
"""
function generate_ragchew_extras(rng::AbstractRNG, station::Station, ::CasualRagchewer)
    phrases = String[]
    rand(rng) < 0.8 && push!(phrases, "NAME $(station.name) $(station.name)")
    rand(rng) < 0.7 && push!(phrases, "QTH $(station.qth)")
    rand(rng) < 0.4 && push!(phrases, "RIG $(rand(rng, ["FT991","IC7300","TS590","K3","FTDX10","IC7610"]))")
    rand(rng) < 0.3 && push!(phrases, "ANT $(rand(rng, ["DIPOLE","GP","YAGI","LOOP","INVV","BEAM"]))")
    rand(rng) < 0.3 && push!(phrases, "WX $(rand(rng, ["SUNNY","CLOUDY","RAIN","SNOW","COLD","WARM","HOT"]))")
    rand(rng) < 0.2 && push!(phrases, "PWR $(rand(rng, ["100W","5W","50W","400W","1KW"]))")
    return phrases
end

function generate_ragchew_extras(rng::AbstractRNG, station::Station, ::AbstractOperatorStyle)
    # Non-ragchew styles produce minimal extras
    phrases = String[]
    rand(rng) < verbosity(station.style) && push!(phrases, "NAME $(station.name)")
    return phrases
end

# ============================================================================
# Error / Correction Simulation
# ============================================================================

"""
    maybe_insert_error(rng, text, style) -> String

Probabilistically introduce sending errors and corrections.
"""
function maybe_insert_error(rng::AbstractRNG, text::String, style::AbstractOperatorStyle)
    words = split(text)
    result = String[]
    for word in words
        if rand(rng) < error_rate(style) && length(word) > 1
            # Simulate sending error then correction
            corrupted = corrupt_word(rng, word)
            push!(result, corrupted)
            # Insert correction marker
            push!(result, rand(rng, Bool) ? "EEE" : "CORR")
            push!(result, String(word))
        else
            push!(result, String(word))
        end
    end
    return join(result, " ")
end

"""
    corrupt_word(rng, word) -> String

Create a corrupted version of a word (missing/swapped characters).
"""
function corrupt_word(rng::AbstractRNG, word::AbstractString)
    chars = collect(word)
    n = length(chars)
    n < 2 && return String(chars)
    action = rand(rng, 1:3)
    if action == 1 && n > 2
        # Drop a character
        idx = rand(rng, 1:n)
        deleteat!(chars, idx)
    elseif action == 2
        # Swap adjacent characters
        idx = rand(rng, 1:n-1)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    else
        # Replace a character
        idx = rand(rng, 1:n)
        chars[idx] = rand(rng, 'A':'Z')
    end
    return String(chars)
end

# ============================================================================
# Retry / AGN Phrases
# ============================================================================

# With this probability, a retry phrase includes a partial callsign (?7NA, NA?, etc.)
const PARTIAL_CALLSIGN_IN_RETRY_PROB = 0.15

"""
    partial_callsign_for_retry(rng, callsign) -> String

Generate a partial callsign as used when asking for a repeat (e.g. ?7NA, NA?, 7NA?).
Natural in AGN/PSE AGN context when the operator didn't copy the full call.
"""
function partial_callsign_for_retry(rng::AbstractRNG, callsign::AbstractString)
    isempty(callsign) && return "?"
    n = length(callsign)
    n <= 2 && return callsign * "?"
    # 2–3 char suffix or prefix with ?; sometimes middle fragment
    choice = rand(rng, 1:3)
    if choice == 1
        # "?7NA" — leading ? + last 2–3 chars
        len = rand(rng, 2:min(3, n))
        "?" * callsign[max(1, n - len + 1):end]
    elseif choice == 2
        # "NA?" — last 2–3 chars + trailing ?
        len = rand(rng, 2:min(3, n))
        callsign[max(1, n - len + 1):end] * "?"
    else
        # "7NA?" — suffix with trailing ?
        len = rand(rng, 2:min(3, n))
        callsign[max(1, n - len + 1):end] * "?"
    end
end

function _maybe_append_partial_retry(rng::AbstractRNG, station::Station, base::String)
    if rand(rng) < PARTIAL_CALLSIGN_IN_RETRY_PROB && !isempty(station.qso_partner)
        return base * " " * partial_callsign_for_retry(rng, station.qso_partner)
    end
    return base
end

"""
    generate_retry(rng, station, style) -> String

Generate a retry/repeat request. With small probability includes a partial
callsign (?7NA, NA?, etc.) as part of the request.
"""
function generate_retry(rng::AbstractRNG, station::Station, ::FastContestOp)
    base = rand(rng, ["AGN", "?", "$(station.qso_partner)?"])
    return _maybe_append_partial_retry(rng, station, base)
end

function generate_retry(rng::AbstractRNG, station::Station, ::CasualRagchewer)
    base = rand(rng, ["PSE AGN", "AGN PSE", "QRM PSE RPT", "AGN?", "SRI AGN"])
    return _maybe_append_partial_retry(rng, station, base)
end

function generate_retry(rng::AbstractRNG, station::Station, ::BeginnerOp)
    base = rand(rng, ["PSE AGN AGN", "PSE RPT UR RST", "?", "AGN PSE AGN"])
    return _maybe_append_partial_retry(rng, station, base)
end

function generate_retry(rng::AbstractRNG, station::Station, ::DXPileupManager)
    partial = station.qso_partner[max(1,end-2):end]
    base = rand(rng, ["$(partial)?", "AGN", "?"])
    return _maybe_append_partial_retry(rng, station, base)
end

function generate_retry(rng::AbstractRNG, station::Station, ::MidSkillContestOp)
    base = rand(rng, ["AGN", "AGN?", "NR AGN", "?"])
    return _maybe_append_partial_retry(rng, station, base)
end

function generate_retry(rng::AbstractRNG, station::Station, ::QRPOperator)
    base = rand(rng, ["PSE AGN", "QRS PSE", "AGN AGN", "RPT PSE"])
    return _maybe_append_partial_retry(rng, station, base)
end
