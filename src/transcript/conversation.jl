"""
    MorseSimulator.jl/src/transcript/conversation.jl

Conversation engine generating realistic CW transcripts.
Uses dispatch on conversation mode to produce different
interaction patterns (contest, ragchew, pileup, test).
"""

using Random, Distributions

# ============================================================================
# Main Entry Point
# ============================================================================

"""
    generate_transcript(rng; kwargs...) -> Transcript

Generate a complete CW transcript with random parameters.

# Keyword Arguments
- `num_stations::Int` — number of stations (2-6)
- `contest` — contest type (random if omitted)
- `mode` — conversation mode (random if omitted)
- `propagation` — propagation conditions
"""
function generate_transcript(rng::AbstractRNG; kwargs...)
    scene = BandScene(rng; kwargs...)
    return generate_transcript(rng, scene)
end

generate_transcript(; kwargs...) = generate_transcript(Random.default_rng(); kwargs...)

"""
    generate_transcript(rng, scene) -> Transcript

Generate a transcript for the given band scene.
Dispatches on scene.mode for mode-specific conversation flow.
"""
function generate_transcript(rng::AbstractRNG, scene::BandScene)
    generate_conversation(rng, scene, scene.mode)
end

# ============================================================================
# Contest Mode Conversation
# ============================================================================

"""
    generate_conversation(rng, scene, ::ContestMode) -> Transcript

Simulate a contest operating session.
One station calls CQ, others respond, exchanges happen.
"""
function generate_conversation(rng::AbstractRNG, scene::BandScene, ::ContestMode)
    txs = Transmission[]
    t = 0.0

    stations = scene.stations
    n = length(stations)
    # Station 1 is the "runner" (calling CQ)
    runner = stations[1]
    callers = stations[2:end]

    # Generate multiple QSOs
    num_qsos = rand(rng, 1:min(n-1, 4))

    for qso_idx in 1:num_qsos
        caller = callers[mod1(qso_idx, length(callers))]
        serial_r = advance_serial!(runner)
        serial_c = advance_serial!(caller)

        # Runner calls CQ
        cq_text = generate_cq(rng, runner, runner.style)
        cq_text = maybe_insert_error(rng, cq_text, runner.style)
        wpm_r = instantaneous_wpm(rng, runner)
        dur = estimate_duration(cq_text, wpm_r)
        sig_r = signal_strength(rng, scene, runner)
        push!(txs, Transmission(runner.callsign, cq_text, t, dur, wpm_r, sig_r))
        t += dur + rand(rng, Uniform(0.3, 1.5))

        # Caller responds
        resp_text = generate_response(rng, caller, runner.callsign, caller.style)
        resp_text = maybe_insert_error(rng, resp_text, caller.style)
        wpm_c = instantaneous_wpm(rng, caller)
        dur = estimate_duration(resp_text, wpm_c)
        sig_c = signal_strength(rng, scene, caller)
        push!(txs, Transmission(caller.callsign, resp_text, t, dur, wpm_c, sig_c))
        t += dur + rand(rng, Uniform(0.2, 1.0))

        # Check for collision: additional station might also call
        if length(callers) > 1 && rand(rng) < scene.collision_probability
            extra = callers[mod1(qso_idx + 1, length(callers))]
            extra_resp = generate_response(rng, extra, runner.callsign, extra.style)
            wpm_e = instantaneous_wpm(rng, extra)
            dur_e = estimate_duration(extra_resp, wpm_e)
            sig_e = signal_strength(rng, scene, extra)
            # Overlapping transmission (slightly earlier)
            overlap_t = t - dur - rand(rng, Uniform(0.0, dur * 0.5))
            push!(txs, Transmission(extra.callsign, extra_resp, overlap_t, dur_e, wpm_e, sig_e))
        end

        # Runner sends exchange
        exch_data = generate_exchange(scene.contest, rng, serial_r)
        exch_text = format_exchange(scene.contest, exch_data)
        exch_phrase = generate_exchange_phrase(rng, runner, caller.callsign, exch_text, runner.style)
        exch_phrase = maybe_insert_error(rng, exch_phrase, runner.style)
        dur = estimate_duration(exch_phrase, wpm_r)
        push!(txs, Transmission(runner.callsign, exch_phrase, t, dur, wpm_r, sig_r))
        t += dur + rand(rng, Uniform(0.2, 0.8))

        # Maybe runner needs to retry (simulating weak signal)
        if rand(rng) < 0.15
            retry_text = generate_retry(rng, caller, caller.style)
            dur = estimate_duration(retry_text, wpm_c)
            push!(txs, Transmission(caller.callsign, retry_text, t, dur, wpm_c, sig_c))
            t += dur + rand(rng, Uniform(0.3, 1.0))

            # Runner resends exchange
            push!(txs, Transmission(runner.callsign, exch_phrase, t, dur, wpm_r, sig_r))
            t += estimate_duration(exch_phrase, wpm_r) + rand(rng, Uniform(0.2, 0.8))
        end

        # Caller sends exchange
        exch_data_c = generate_exchange(scene.contest, rng, serial_c)
        exch_text_c = format_exchange(scene.contest, exch_data_c)
        exch_phrase_c = generate_exchange_phrase(rng, caller, runner.callsign, exch_text_c, caller.style)
        exch_phrase_c = maybe_insert_error(rng, exch_phrase_c, caller.style)
        dur = estimate_duration(exch_phrase_c, wpm_c)
        push!(txs, Transmission(caller.callsign, exch_phrase_c, t, dur, wpm_c, sig_c))
        t += dur + rand(rng, Uniform(0.2, 0.8))

        # Runner confirms
        conf = generate_confirmation(rng, runner, caller.callsign, runner.style)
        dur = estimate_duration(conf, wpm_r)
        push!(txs, Transmission(runner.callsign, conf, t, dur, wpm_r, sig_r))
        t += dur + rand(rng, Uniform(0.3, 1.5))
    end

    label = build_label(txs)
    return Transcript(txs, label, mode_name(ContestMode()), contest_name(scene.contest))
end

# ============================================================================
# Ragchew Mode Conversation
# ============================================================================

function generate_conversation(rng::AbstractRNG, scene::BandScene, ::RagchewMode)
    txs = Transmission[]
    t = 0.0

    stations = scene.stations
    s1 = stations[1]
    s2 = stations[min(2, end)]

    wpm1 = instantaneous_wpm(rng, s1)
    wpm2 = instantaneous_wpm(rng, s2)
    sig1 = signal_strength(rng, scene, s1)
    sig2 = signal_strength(rng, scene, s2)

    # S1 calls CQ
    cq = generate_cq(rng, s1, s1.style)
    dur = estimate_duration(cq, wpm1)
    push!(txs, Transmission(s1.callsign, cq, t, dur, wpm1, sig1))
    t += dur + rand(rng, Uniform(0.5, 2.0))

    # S2 responds
    resp = generate_response(rng, s2, s1.callsign, s2.style)
    dur = estimate_duration(resp, wpm2)
    push!(txs, Transmission(s2.callsign, resp, t, dur, wpm2, sig2))
    t += dur + rand(rng, Uniform(0.5, 1.5))

    # S1 sends RST and extras
    exch = generate_exchange(scene.contest, rng, advance_serial!(s1))
    exch_text = format_exchange(scene.contest, exch)
    greeting = rand(rng, ["GM", "GA", "GE", "GD"])
    base = "$(s2.callsign) DE $(s1.callsign) $greeting OM TNX FER CALL UR RST $exch_text"
    extras1 = generate_ragchew_extras(rng, s1, s1.style)
    full_text = isempty(extras1) ? "$base BK" : "$base $(join(extras1, " ")) BK"
    full_text = maybe_insert_error(rng, full_text, s1.style)
    dur = estimate_duration(full_text, wpm1)
    push!(txs, Transmission(s1.callsign, full_text, t, dur, wpm1, sig1))
    t += dur + rand(rng, Uniform(1.0, 3.0))

    # S2 responds with RST and extras
    exch2 = generate_exchange(scene.contest, rng, advance_serial!(s2))
    exch_text2 = format_exchange(scene.contest, exch2)
    greeting2 = rand(rng, ["R", "R R", "FB"])
    base2 = "$(s1.callsign) DE $(s2.callsign) $greeting2 TNX UR RST $exch_text2"
    extras2 = generate_ragchew_extras(rng, s2, s2.style)
    full_text2 = isempty(extras2) ? "$base2 BK" : "$base2 $(join(extras2, " ")) BK"
    full_text2 = maybe_insert_error(rng, full_text2, s2.style)
    dur = estimate_duration(full_text2, wpm2)
    push!(txs, Transmission(s2.callsign, full_text2, t, dur, wpm2, sig2))
    t += dur + rand(rng, Uniform(1.0, 3.0))

    # S1 closing
    closing = generate_confirmation(rng, s1, s2.callsign, s1.style)
    dur = estimate_duration(closing, wpm1)
    push!(txs, Transmission(s1.callsign, closing, t, dur, wpm1, sig1))
    t += dur + rand(rng, Uniform(0.5, 1.5))

    # S2 closing
    closing2 = "$(s1.callsign) DE $(s2.callsign) 73 SK"
    dur = estimate_duration(closing2, wpm2)
    push!(txs, Transmission(s2.callsign, closing2, t, dur, wpm2, sig2))

    label = build_label(txs)
    return Transcript(txs, label, mode_name(RagchewMode()), contest_name(scene.contest))
end

# ============================================================================
# DX Pileup Mode
# ============================================================================

function generate_conversation(rng::AbstractRNG, scene::BandScene, ::DXPileupMode)
    txs = Transmission[]
    t = 0.0

    stations = scene.stations
    dx = stations[1]  # DX station (pileup manager style)
    callers = stations[2:end]

    wpm_dx = instantaneous_wpm(rng, dx)
    sig_dx = signal_strength(rng, scene, dx)

    # DX calls CQ
    cq = generate_cq(rng, dx, dx.style)
    dur = estimate_duration(cq, wpm_dx)
    push!(txs, Transmission(dx.callsign, cq, t, dur, wpm_dx, sig_dx))
    t += dur + rand(rng, Uniform(0.2, 0.8))

    # Pileup: multiple stations call simultaneously
    num_callers = min(length(callers), rand(rng, Poisson(2.5)))
    num_callers = max(1, min(num_callers, length(callers)))
    pile_callers = callers[1:num_callers]

    # Callers transmit (possibly overlapping)
    for caller in pile_callers
        wpm_c = instantaneous_wpm(rng, caller)
        sig_c = signal_strength(rng, scene, caller)
        resp = caller.callsign  # In pileup, just send callsign
        dur = estimate_duration(resp, wpm_c)
        call_t = t + rand(rng, Uniform(0.0, 0.5))
        push!(txs, Transmission(caller.callsign, resp, call_t, dur, wpm_c, sig_c))
    end
    t += 2.0 + rand(rng, Uniform(0.0, 1.0))

    # DX picks strongest (or random) caller
    picked = pile_callers[rand(rng, 1:length(pile_callers))]
    wpm_p = instantaneous_wpm(rng, picked)
    sig_p = signal_strength(rng, scene, picked)

    # DX responds to picked station
    exch_data = generate_exchange(scene.contest, rng, advance_serial!(dx))
    exch_text = format_exchange(scene.contest, exch_data)
    dx_resp = "$(picked.callsign) $exch_text"
    dur = estimate_duration(dx_resp, wpm_dx)
    push!(txs, Transmission(dx.callsign, dx_resp, t, dur, wpm_dx, sig_dx))
    t += dur + rand(rng, Uniform(0.3, 1.0))

    # Picked station sends exchange
    exch_data_p = generate_exchange(scene.contest, rng, advance_serial!(picked))
    exch_text_p = format_exchange(scene.contest, exch_data_p)
    picked_exch = generate_exchange_phrase(rng, picked, dx.callsign, exch_text_p, picked.style)
    dur = estimate_duration(picked_exch, wpm_p)
    push!(txs, Transmission(picked.callsign, picked_exch, t, dur, wpm_p, sig_p))
    t += dur + rand(rng, Uniform(0.2, 0.6))

    # DX confirms and calls next
    conf = "TU $(dx.callsign)"
    dur = estimate_duration(conf, wpm_dx)
    push!(txs, Transmission(dx.callsign, conf, t, dur, wpm_dx, sig_dx))

    # Sort transmissions by time
    sort!(txs, by = tx -> tx.time_offset)

    label = build_label(txs)
    return Transcript(txs, label, mode_name(DXPileupMode()), contest_name(scene.contest))
end

# ============================================================================
# Test Mode
# ============================================================================

function generate_conversation(rng::AbstractRNG, scene::BandScene, ::TestMode)
    txs = Transmission[]
    t = 0.0

    s1 = scene.stations[1]
    wpm = instantaneous_wpm(rng, s1)
    sig = signal_strength(rng, scene, s1)

    test_type = rand(rng, 1:3)
    text = if test_type == 1
        "VVV VVV VVV DE $(s1.callsign) $(s1.callsign) TEST"
    elseif test_type == 2
        "QRL? QRL? DE $(s1.callsign)"
    else
        patterns = ["THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG",
                    "1 2 3 4 5 6 7 8 9 0",
                    "CQ CQ CQ DE $(s1.callsign) $(s1.callsign) TEST TEST K"]
        rand(rng, patterns)
    end

    dur = estimate_duration(text, wpm)
    push!(txs, Transmission(s1.callsign, text, t, dur, wpm, sig))

    label = build_label(txs)
    return Transcript(txs, label, mode_name(TestMode()), contest_name(scene.contest))
end

# ============================================================================
# Label Builder
# ============================================================================

"""
    build_label(txs) -> String

Build the flat training label from transmissions.
Format: <START> text text text <END>
"""
function build_label(txs::Vector{Transmission})
    sorted = sort(txs, by = tx -> tx.time_offset)
    parts = [tx.text for tx in sorted]
    return "<START> " * join(parts, " ") * " <END>"
end

# ============================================================================
# Batch Generation
# ============================================================================

"""
    generate_transcripts(rng, n; kwargs...) -> Vector{Transcript}

Generate `n` diverse transcripts with random parameters.
"""
function generate_transcripts(rng::AbstractRNG, n::Int; kwargs...)
    return [generate_transcript(rng; kwargs...) for _ in 1:n]
end

generate_transcripts(n::Int; kwargs...) =
    generate_transcripts(Random.default_rng(), n; kwargs...)
