"""
    MorseSimulator.jl/src/transcript/callsigns.jl

Realistic amateur radio callsign generation with proper prefix/suffix
patterns organized by ITU region and country.
"""

using Random

"""
    CallsignPrefix

A callsign prefix pattern associated with a country/region.
"""
struct CallsignPrefix
    prefix::String
    country::String
    region::Int  # ITU region 1, 2, or 3
end

"""
    CALLSIGN_PREFIXES

Table of common amateur radio prefixes by country.
"""
const CALLSIGN_PREFIXES = CallsignPrefix[
    # Region 1 - Europe, Africa, Middle East
    CallsignPrefix("SP", "Poland", 1),
    CallsignPrefix("SO", "Poland", 1),
    CallsignPrefix("SQ", "Poland", 1),
    CallsignPrefix("SN", "Poland", 1),
    CallsignPrefix("DL", "Germany", 1),
    CallsignPrefix("DJ", "Germany", 1),
    CallsignPrefix("DK", "Germany", 1),
    CallsignPrefix("G",  "England", 1),
    CallsignPrefix("M",  "England", 1),
    CallsignPrefix("F",  "France", 1),
    CallsignPrefix("ON", "Belgium", 1),
    CallsignPrefix("PA", "Netherlands", 1),
    CallsignPrefix("I",  "Italy", 1),
    CallsignPrefix("IK", "Italy", 1),
    CallsignPrefix("EA", "Spain", 1),
    CallsignPrefix("OK", "Czech Republic", 1),
    CallsignPrefix("OM", "Slovakia", 1),
    CallsignPrefix("HA", "Hungary", 1),
    CallsignPrefix("YU", "Serbia", 1),
    CallsignPrefix("LZ", "Bulgaria", 1),
    CallsignPrefix("UR", "Ukraine", 1),
    CallsignPrefix("UA", "Russia", 1),
    CallsignPrefix("OH", "Finland", 1),
    CallsignPrefix("SM", "Sweden", 1),
    CallsignPrefix("LA", "Norway", 1),
    CallsignPrefix("OZ", "Denmark", 1),
    CallsignPrefix("9A", "Croatia", 1),
    CallsignPrefix("S5", "Slovenia", 1),
    CallsignPrefix("YO", "Romania", 1),
    # Region 2 - Americas
    CallsignPrefix("W",  "USA", 2),
    CallsignPrefix("K",  "USA", 2),
    CallsignPrefix("N",  "USA", 2),
    CallsignPrefix("VE", "Canada", 2),
    CallsignPrefix("VA", "Canada", 2),
    CallsignPrefix("XE", "Mexico", 2),
    CallsignPrefix("LU", "Argentina", 2),
    CallsignPrefix("PY", "Brazil", 2),
    CallsignPrefix("CE", "Chile", 2),
    CallsignPrefix("HK", "Colombia", 2),
    # Region 3 - Asia, Pacific
    CallsignPrefix("JA", "Japan", 3),
    CallsignPrefix("JH", "Japan", 3),
    CallsignPrefix("BV", "Taiwan", 3),
    CallsignPrefix("HL", "South Korea", 3),
    CallsignPrefix("VK", "Australia", 3),
    CallsignPrefix("ZL", "New Zealand", 3),
    CallsignPrefix("VU", "India", 3),
    CallsignPrefix("BY", "China", 3),
    CallsignPrefix("DU", "Philippines", 3),
    CallsignPrefix("HS", "Thailand", 3),
]

"""
    generate_callsign(rng::AbstractRNG) -> String

Generate a random but realistic amateur radio callsign.
Format: PREFIX + DIGIT + SUFFIX (1-3 letters)
"""
function generate_callsign(rng::AbstractRNG)
    entry = CALLSIGN_PREFIXES[rand(rng, 1:length(CALLSIGN_PREFIXES))]
    digit = rand(rng, 0:9)
    suffix_len = rand(rng, 1:3)
    suffix = String([rand(rng, 'A':'Z') for _ in 1:suffix_len])
    return "$(entry.prefix)$(digit)$(suffix)"
end

"""
    generate_callsign() -> String

Generate a random callsign using the global RNG.
"""
generate_callsign() = generate_callsign(Random.default_rng())

"""
    generate_callsigns(rng::AbstractRNG, n::Int) -> Vector{String}

Generate `n` unique callsigns.
"""
function generate_callsigns(rng::AbstractRNG, n::Int)
    calls = Set{String}()
    while length(calls) < n
        push!(calls, generate_callsign(rng))
    end
    return collect(calls)
end

generate_callsigns(n::Int) = generate_callsigns(Random.default_rng(), n)

"""
    callsign_region(call::AbstractString) -> Int

Determine ITU region from callsign prefix (heuristic).
"""
function callsign_region(call::AbstractString)
    for entry in CALLSIGN_PREFIXES
        if startswith(call, entry.prefix)
            return entry.region
        end
    end
    return 1  # default
end
