"""
    CWContestSim/src/morse/code.jl

International Morse Code table and character-to-symbol conversion.
Uses a compile-time constant dictionary for fast lookup.
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

const _D = Dot()
const _H = Dash()

"""
    MORSE_TABLE

Mapping from characters to Morse symbol sequences.
"""
const MORSE_TABLE = Dict{Char, Vector{AbstractMorseElement}}(
    # Letters
    'A' => [_D, _H],
    'B' => [_H, _D, _D, _D],
    'C' => [_H, _D, _H, _D],
    'D' => [_H, _D, _D],
    'E' => [_D],
    'F' => [_D, _D, _H, _D],
    'G' => [_H, _H, _D],
    'H' => [_D, _D, _D, _D],
    'I' => [_D, _D],
    'J' => [_D, _H, _H, _H],
    'K' => [_H, _D, _H],
    'L' => [_D, _H, _D, _D],
    'M' => [_H, _H],
    'N' => [_H, _D],
    'O' => [_H, _H, _H],
    'P' => [_D, _H, _H, _D],
    'Q' => [_H, _H, _D, _H],
    'R' => [_D, _H, _D],
    'S' => [_D, _D, _D],
    'T' => [_H],
    'U' => [_D, _D, _H],
    'V' => [_D, _D, _D, _H],
    'W' => [_D, _H, _H],
    'X' => [_H, _D, _D, _H],
    'Y' => [_H, _D, _H, _H],
    'Z' => [_H, _H, _D, _D],
    # Numbers
    '0' => [_H, _H, _H, _H, _H],
    '1' => [_D, _H, _H, _H, _H],
    '2' => [_D, _D, _H, _H, _H],
    '3' => [_D, _D, _D, _H, _H],
    '4' => [_D, _D, _D, _D, _H],
    '5' => [_D, _D, _D, _D, _D],
    '6' => [_H, _D, _D, _D, _D],
    '7' => [_H, _H, _D, _D, _D],
    '8' => [_H, _H, _H, _D, _D],
    '9' => [_H, _H, _H, _H, _D],
    # Punctuation
    '.' => [_D, _H, _D, _H, _D, _H],
    ',' => [_H, _H, _D, _D, _H, _H],
    '?' => [_D, _D, _H, _H, _D, _D],
    '/' => [_H, _D, _D, _H, _D],
    '=' => [_H, _D, _D, _D, _H],
    '+' => [_D, _H, _D, _H, _D],
    '-' => [_H, _D, _D, _D, _D, _H],
)

# Prosigns (sent as single characters without inter-character gap)
const PROSIGN_TABLE = Dict{String, Vector{AbstractMorseElement}}(
    "AR" => [_D, _H, _D, _H, _D],
    "SK" => [_D, _D, _D, _H, _D, _H],
    "BT" => [_H, _D, _D, _D, _H],
    "KN" => [_H, _D, _H, _H, _D],
    "BK" => [_H, _D, _D, _D, _H, _D, _H],
    "EEE"=> [_D, _D, _D],  # Error signal
)

"""
    char_to_morse(c::Char) -> Vector{AbstractMorseElement}

Convert a single character to its Morse symbol sequence.
Returns empty vector for unknown characters.
"""
function char_to_morse(c::Char)
    uc = uppercase(c)
    return get(MORSE_TABLE, uc, AbstractMorseElement[])
end

"""
    is_prosign(word::AbstractString) -> Bool

Check if a word is a prosign.
"""
function is_prosign(word::AbstractString)
    return haskey(PROSIGN_TABLE, uppercase(String(word)))
end

"""
    prosign_to_morse(word::AbstractString) -> Vector{AbstractMorseElement}

Convert a prosign to its Morse symbol sequence.
"""
function prosign_to_morse(word::AbstractString)
    return get(PROSIGN_TABLE, uppercase(String(word)), AbstractMorseElement[])
end
