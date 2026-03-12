"""
    MorseSimulator.jl/src/transcript/bandscene.jl

Band scene model representing the radio environment in which
stations operate. Controls propagation, noise, fading, and
station density parameters.
"""

using Random, Distributions

"""
    PropagationCondition

Propagation conditions affecting signal strength and fading.
"""
struct PropagationCondition
    path_loss_mean::Float64      # Mean path loss in dB
    path_loss_sigma::Float64     # Variability of path loss
    qsb_probability::Float64    # Probability of QSB fading
    qsb_depth_mean::Float64     # Mean fading depth (0-1)
    qsb_freq_range::Tuple{Float64,Float64}  # QSB frequency range (Hz)
end

"""
    good_propagation() -> PropagationCondition
"""
good_propagation() = PropagationCondition(20.0, 5.0, 0.2, 0.1, (0.05, 0.3))

"""
    moderate_propagation() -> PropagationCondition
"""
moderate_propagation() = PropagationCondition(40.0, 10.0, 0.5, 0.3, (0.1, 0.5))

"""
    poor_propagation() -> PropagationCondition
"""
poor_propagation() = PropagationCondition(60.0, 15.0, 0.8, 0.6, (0.2, 1.0))

"""
    BandScene{C<:AbstractContest, M<:AbstractConversationMode}

The simulated radio band environment.

# Fields
- `contest::C` — active contest type
- `mode::M` — conversation mode
- `propagation::PropagationCondition` — propagation conditions
- `noise_floor_db::Float64` — band noise floor in dB
- `station_density::Float64` — expected number of active stations
- `collision_probability::Float64` — probability of signal overlap
- `stations::Vector{Station}` — stations present on the band
"""
struct BandScene{C<:AbstractContest, M<:AbstractConversationMode}
    contest::C
    mode::M
    propagation::PropagationCondition
    noise_floor_db::Float64
    station_density::Float64
    collision_probability::Float64
    stations::Vector{<:Station}
end

"""
    BandScene(rng; num_stations, contest, mode, propagation, noise_floor_db) -> BandScene

Construct a BandScene with the given or random parameters.
"""
function BandScene(rng::AbstractRNG;
                   num_stations::Int = rand(rng, 2:6),
                   contest::AbstractContest = random_contest(rng),
                   mode::AbstractConversationMode = random_mode(rng),
                   propagation::PropagationCondition = moderate_propagation(),
                   noise_floor_db::Float64 = -28.0 + rand(rng) * 16.0)  # -28 to -12 dB; SNR emerges from signal levels

    callsigns = generate_callsigns(rng, num_stations)
    styles = select_styles(rng, mode, num_stations)
    # Caller tone: most operators 400–700 Hz, mean ~600; full 200–900 Hz inspected in mel for outliers
    ref_freq = clamp(600.0 + 80.0 * randn(rng), 400.0, 700.0)
    stations = [
        Station(rng, call, style; ref_freq=ref_freq, is_caller=(i == 1))
        for (i, (call, style)) in enumerate(zip(callsigns, styles))
    ]

    density = Float64(num_stations)
    collision_p = clamp(0.05 * num_stations, 0.0, 0.6)

    BandScene(contest, mode, propagation, noise_floor_db,
              density, collision_p, stations)
end

"""
    select_styles(rng, mode, n) -> Vector{AbstractOperatorStyle}

Select operator styles appropriate for the conversation mode.
"""
function select_styles(rng::AbstractRNG, ::ContestMode, n::Int)
    pool = [FastContestOp(), MidSkillContestOp(), QRPOperator(), BeginnerOp()]
    return [pool[rand(rng, 1:length(pool))] for _ in 1:n]
end

function select_styles(rng::AbstractRNG, ::RagchewMode, n::Int)
    pool = [CasualRagchewer(), MidSkillContestOp(), QRPOperator(), BeginnerOp()]
    return [pool[rand(rng, 1:length(pool))] for _ in 1:n]
end

function select_styles(rng::AbstractRNG, ::DXPileupMode, n::Int)
    styles = AbstractOperatorStyle[DXPileupManager()]
    callers = [FastContestOp(), MidSkillContestOp(), QRPOperator(), CasualRagchewer()]
    for _ in 2:n
        push!(styles, callers[rand(rng, 1:length(callers))])
    end
    return styles
end

function select_styles(rng::AbstractRNG, ::TestMode, n::Int)
    pool = [BeginnerOp(), CasualRagchewer(), MidSkillContestOp()]
    return [pool[rand(rng, 1:length(pool))] for _ in 1:n]
end

"""
    signal_strength(rng, scene, station) -> Float64

Compute received signal strength for a station considering
propagation and station amplitude.
"""
function signal_strength(rng::AbstractRNG, scene::BandScene, station::Station)
    path_loss = scene.propagation.path_loss_mean + randn(rng) * scene.propagation.path_loss_sigma
    raw = station.signal_amplitude * 10.0^(-path_loss / 20.0)
    return max(0.001, raw)
end
