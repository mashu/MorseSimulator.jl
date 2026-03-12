"""
    CWContestSim/src/visualization/plotting.jl

Visualization utilities using UnicodePlots for terminal-based
inspection of transcripts, Morse timing, and mel-spectrograms.
"""

using UnicodePlots

# ============================================================================
# Transcript Visualization
# ============================================================================

"""
    plot_transcript(transcript; width) -> Nothing

Display a visual timeline of transmissions in a transcript.
"""
function plot_transcript(transcript::Transcript; width::Int = 80)
    txs = transcript.transmissions
    isempty(txs) && (println("Empty transcript"); return nothing)

    println("━" ^ width)
    println("  Mode: $(transcript.mode_name) │ Contest: $(transcript.contest_name)")
    println("━" ^ width)

    max_time = maximum(tx.time_offset + tx.duration_estimate for tx in txs)
    max_time = max(max_time, 0.1)

    # Assign colors by callsign
    callsigns = unique(tx.callsign for tx in txs)
    colors = [:cyan, :yellow, :green, :magenta, :red, :blue]

    for tx in txs
        cidx = findfirst(==(tx.callsign), callsigns)
        color = colors[mod1(cidx, length(colors))]

        # Time bar
        start_pos = round(Int, tx.time_offset / max_time * (width - 20)) + 1
        end_pos = round(Int, (tx.time_offset + tx.duration_estimate) / max_time * (width - 20)) + 1
        start_pos = clamp(start_pos, 1, width - 20)
        end_pos = clamp(end_pos, start_pos, width - 20)

        bar = " " ^ (start_pos - 1) * "█" ^ max(1, end_pos - start_pos)

        t_str = lpad("$(round(tx.time_offset, digits=1))s", 6)
        call_str = rpad(tx.callsign, 8)
        println("  $t_str │ $call_str │ $bar")
        # Truncated text
        txt = length(tx.text) > width - 25 ? tx.text[1:width-28] * "..." : tx.text
        println("         │          │ $txt")
    end

    println("━" ^ width)
    println("  Label: $(transcript.label[1:min(end, width-10)])")
    println("━" ^ width)

    return nothing
end

# ============================================================================
# Spectrogram Visualization
# ============================================================================

"""
    plot_spectrogram(result; title, width, height) -> Plot

Display a mel-spectrogram using UnicodePlots heatmap.
"""
function plot_spectrogram(result::SpectrogramResult{T};
                          title::String = "Mel Spectrogram",
                          width::Int = 80,
                          height::Int = 20) where T<:AbstractFloat
    spec = result.mel_spectrogram

    # Subsample if too wide for terminal
    n_frames = size(spec, 2)
    n_mels = size(spec, 1)

    if n_frames > width * 2
        step = n_frames ÷ (width * 2)
        spec = spec[:, 1:step:end]
    end

    plt = heatmap(spec;
        title = title,
        xlabel = "Time (frames)",
        ylabel = "Mel bin",
        width = width,
        height = height,
        colormap = :inferno
    )

    return plt
end

"""
    plot_spectrogram(spec::Matrix; kwargs...)

Plot a raw spectrogram matrix.
"""
function plot_spectrogram(spec::Matrix{T};
                          title::String = "Spectrogram",
                          width::Int = 80,
                          height::Int = 20) where T<:AbstractFloat
    n_frames = size(spec, 2)
    if n_frames > width * 2
        step = n_frames ÷ (width * 2)
        spec = spec[:, 1:step:end]
    end

    plt = heatmap(spec;
        title = title,
        xlabel = "Time (frames)",
        ylabel = "Frequency bin",
        width = width,
        height = height,
        colormap = :inferno
    )

    return plt
end

# ============================================================================
# Waveform Visualization
# ============================================================================

"""
    plot_waveform(signal; title, width, height) -> Plot

Display a waveform using UnicodePlots lineplot.
"""
function plot_waveform(signal::MixedSignal{T};
                       title::String = "CW Signal",
                       width::Int = 80,
                       height::Int = 15,
                       max_samples::Int = 5000) where T<:AbstractFloat
    samples = signal.samples
    n = length(samples)

    # Subsample for display
    if n > max_samples
        step = n ÷ max_samples
        samples = samples[1:step:end]
    end

    times = range(0, signal.duration, length=length(samples))

    plt = lineplot(collect(times), collect(samples);
        title = title,
        xlabel = "Time (s)",
        ylabel = "Amplitude",
        width = width,
        height = height
    )

    return plt
end

"""
    plot_waveform(samples::Vector, sample_rate::Int; kwargs...)

Plot raw audio samples.
"""
function plot_waveform(samples::Vector{T}, sample_rate::Int;
                       title::String = "Waveform",
                       width::Int = 80,
                       height::Int = 15,
                       max_samples::Int = 5000) where T<:AbstractFloat
    n = length(samples)
    if n > max_samples
        step = n ÷ max_samples
        samples = samples[1:step:end]
    end

    duration = n / sample_rate
    times = range(0, duration, length=length(samples))

    plt = lineplot(collect(times), collect(samples);
        title = title,
        xlabel = "Time (s)",
        ylabel = "Amplitude",
        width = width,
        height = height
    )

    return plt
end

# ============================================================================
# Morse Timing Visualization
# ============================================================================

"""
    plot_morse_timing(events; title, width, height, max_events) -> Plot

Visualize Morse timing events as a keying waveform.
"""
function plot_morse_timing(events::Vector{TimedMorseEvent{T}};
                           title::String = "Morse Keying",
                           width::Int = 80,
                           height::Int = 8,
                           max_events::Int = 200) where T<:AbstractFloat
    isempty(events) && return nothing

    # Build step waveform
    times = T[]
    levels = T[]

    evs = length(events) > max_events ? events[1:max_events] : events

    for event in evs
        push!(times, event.start_time)
        push!(levels, is_keyed(event.element) ? one(T) : zero(T))
        push!(times, event.start_time + event.duration)
        push!(levels, is_keyed(event.element) ? one(T) : zero(T))
    end

    plt = lineplot(times, levels;
        title = title,
        xlabel = "Time (s)",
        ylabel = "Key",
        width = width,
        height = height,
        ylim = (T(-0.1), T(1.1))
    )

    return plt
end

# ============================================================================
# Consistency Report Visualization
# ============================================================================

"""
    plot_consistency(report) -> Nothing

Print a formatted consistency report.
"""
function plot_consistency(report::ConsistencyReport)
    println("┌────────────────────────────────────────────┐")
    println("│  Path Consistency Comparison               │")
    println("├────────────────────────────────────────────┤")
    println("│  L2 Error:     $(lpad(round(report.l2_error, digits=6), 12))     │")
    println("│  Cosine Sim:   $(lpad(round(report.cosine_similarity, digits=6), 12))     │")
    println("│  KL Divergence:$(lpad(round(report.kl_divergence, digits=6), 12))     │")
    println("│  MAE:          $(lpad(round(report.mae, digits=6), 12))     │")
    println("└────────────────────────────────────────────┘")
    return nothing
end
