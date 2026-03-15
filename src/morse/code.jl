"""
    MorseSimulator.jl/src/morse/code.jl

International Morse Code table and character-to-symbol conversion.
Uses a compile-time constant dictionary for fast lookup.

Type stability: MORSE_TABLE and PROSIGN_TABLE use `Vector{DotOrDash}` (a small
concrete Union that Julia can split efficiently) instead of `Vector{AbstractMorseElement}`.
The full 5-type union `MorseElement` is used in `TimedMorseEvent` and dispatch contexts.
"""

# ============================================================================
# Morse Elements
# ============================================================================

"""
    Dot <: AbstractMorseElement

A Morse dot (dit). Duration = 1 time unit.
"""
struct Dot <: AbstractMorseElement end

"""
    Dash <: AbstractMorseElement

A Morse dash (dah). Duration = 3 time units.
"""
struct Dash <: AbstractMorseElement end

"""
    SymbolGap <: AbstractMorseElement

Gap between symbols within a character. Duration = 1 time unit.
"""
struct SymbolGap <: AbstractMorseElement end

"""
    CharGap <: AbstractMorseElement

Gap between characters. Duration = 3 time units.
"""
struct CharGap <: AbstractMorseElement end

"""
    WordGap <: AbstractMorseElement

Gap between words. Duration = 7 time units.
"""
struct WordGap <: AbstractMorseElement end

# ── Concrete union aliases for type-stable containers ────────────────────────
# Julia's small-union optimization handles these efficiently at compile time.

"""Union of keying elements (Dot/Dash only) — used in MORSE_TABLE / PROSIGN_TABLE."""
const DotOrDash = Union{Dot, Dash}

"""Full union of all five Morse elements — used in TimedMorseEvent and dispatch."""
const MorseElement = Union{Dot, Dash, SymbolGap, CharGap, WordGap}

# Duration in dot-units via dispatch
"""
    dot_units(element) -> Int

Duration of a Morse element in dot units.
"""
dot_units(::Dot)       = 1
dot_units(::Dash)      = 3
dot_units(::SymbolGap) = 1
dot_units(::CharGap)   = 3
dot_units(::WordGap)   = 7

# Is this element a keyed (tone-on) element?
"""
    is_keyed(element) -> Bool

Returns true if the element produces a tone (dot or dash).
"""
is_keyed(::Dot)        = true
is_keyed(::Dash)       = true
is_keyed(::SymbolGap)  = false
is_keyed(::CharGap)    = false
is_keyed(::WordGap)    = false

# ============================================================================
# Morse Code Table
# ============================================================================

# Module-level shorthand for table construction (not exported, not underscore-prefixed)
const DIT = Dot()
const DAH = Dash()

"""
    MORSE_TABLE

Mapping from characters to Morse symbol sequences (Dot/Dash only).
"""
const MORSE_TABLE = Dict{Char, Vector{DotOrDash}}(
    # Letters
    'A' => [DIT, DAH],
    'B' => [DAH, DIT, DIT, DIT],
    'C' => [DAH, DIT, DAH, DIT],
    'D' => [DAH, DIT, DIT],
    'E' => [DIT],
    'F' => [DIT, DIT, DAH, DIT],
    'G' => [DAH, DAH, DIT],
    'H' => [DIT, DIT, DIT, DIT],
    'I' => [DIT, DIT],
    'J' => [DIT, DAH, DAH, DAH],
    'K' => [DAH, DIT, DAH],
    'L' => [DIT, DAH, DIT, DIT],
    'M' => [DAH, DAH],
    'N' => [DAH, DIT],
    'O' => [DAH, DAH, DAH],
    'P' => [DIT, DAH, DAH, DIT],
    'Q' => [DAH, DAH, DIT, DAH],
    'R' => [DIT, DAH, DIT],
    'S' => [DIT, DIT, DIT],
    'T' => [DAH],
    'U' => [DIT, DIT, DAH],
    'V' => [DIT, DIT, DIT, DAH],
    'W' => [DIT, DAH, DAH],
    'X' => [DAH, DIT, DIT, DAH],
    'Y' => [DAH, DIT, DAH, DAH],
    'Z' => [DAH, DAH, DIT, DIT],
    # Numbers
    '0' => [DAH, DAH, DAH, DAH, DAH],
    '1' => [DIT, DAH, DAH, DAH, DAH],
    '2' => [DIT, DIT, DAH, DAH, DAH],
    '3' => [DIT, DIT, DIT, DAH, DAH],
    '4' => [DIT, DIT, DIT, DIT, DAH],
    '5' => [DIT, DIT, DIT, DIT, DIT],
    '6' => [DAH, DIT, DIT, DIT, DIT],
    '7' => [DAH, DAH, DIT, DIT, DIT],
    '8' => [DAH, DAH, DAH, DIT, DIT],
    '9' => [DAH, DAH, DAH, DAH, DIT],
    # Punctuation
    '.' => [DIT, DAH, DIT, DAH, DIT, DAH],
    ',' => [DAH, DAH, DIT, DIT, DAH, DAH],
    '?' => [DIT, DIT, DAH, DAH, DIT, DIT],
    '/' => [DAH, DIT, DIT, DAH, DIT],
    '=' => [DAH, DIT, DIT, DIT, DAH],
    '+' => [DIT, DAH, DIT, DAH, DIT],
    '-' => [DAH, DIT, DIT, DIT, DIT, DAH],
)

# Prosigns (sent as single characters without inter-character gap)
const PROSIGN_TABLE = Dict{String, Vector{DotOrDash}}(
    "AR" => [DIT, DAH, DIT, DAH, DIT],
    "SK" => [DIT, DIT, DIT, DAH, DIT, DAH],
    "BT" => [DAH, DIT, DIT, DIT, DAH],
    "KN" => [DAH, DIT, DAH, DAH, DIT],
    "BK" => [DAH, DIT, DIT, DIT, DAH, DIT, DAH],
    "EEE"=> [DIT, DIT, DIT],  # Error signal
)

"""
    char_to_morse(c::Char) -> Vector{DotOrDash}

Convert a single character to its Morse symbol sequence.
Returns empty vector for unknown characters.
"""
function char_to_morse(c::Char)
    uc = uppercase(c)
    return get(MORSE_TABLE, uc, DotOrDash[])
end

"""
    is_prosign(word::AbstractString) -> Bool

Check if a word is a prosign.
"""
function is_prosign(word::AbstractString)
    return haskey(PROSIGN_TABLE, uppercase(String(word)))
end

"""
    prosign_to_morse(word::AbstractString) -> Vector{DotOrDash}

Convert a prosign to its Morse symbol sequence.
"""
function prosign_to_morse(word::AbstractString)
    return get(PROSIGN_TABLE, uppercase(String(word)), DotOrDash[])
end
