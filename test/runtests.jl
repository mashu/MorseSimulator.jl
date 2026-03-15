"""
    MorseSimulator/test/runtests.jl

Test suite covering transcript generation, Morse encoding,
signal generation, and spectrogram computation.
"""

using Test
using Random
using MorseSimulator

const RNG = MersenneTwister(12345)

@testset "MorseSimulator" begin

    @testset "Callsign Generation" begin
        call = generate_callsign(RNG)
        @test length(call) >= 3
        @test all(c -> isuppercase(c) || isdigit(c), call)

        calls = generate_callsigns(RNG, 10)
        @test length(calls) == 10
        @test length(unique(calls)) == 10
    end

    @testset "Contest Exchange Generation" begin
        for contest in ALL_CONTESTS
            exch = generate_exchange(contest, RNG, 1)
            text = format_exchange(contest, exch)
            @test length(text) > 0
            @test length(contest_name(contest)) > 0
        end
    end

    @testset "Operator Styles" begin
        for style in ALL_STYLES
            @test mean_wpm(style) > 0
            @test wpm_sigma(style) >= 0
            @test 0.0 <= verbosity(style) <= 1.0
            @test patience(style) > 0
            @test 0.0 <= error_rate(style) <= 1.0
            @test 0.0 <= aggressiveness(style) <= 1.0
        end
    end

    @testset "Station Construction" begin
        station = Station(RNG, "W1ABC", FastContestOp())
        @test station.callsign == "W1ABC"
        @test station.tone_freq >= 400.0
        @test station.tone_freq <= 900.0

        wpm = instantaneous_wpm(RNG, station)
        @test wpm > 5.0

        serial = advance_serial!(station)
        @test serial == 1
        serial2 = advance_serial!(station)
        @test serial2 == 2
    end

    @testset "Transcript Generation" begin
        for mode in [ContestMode(), RagchewMode(), DXPileupMode(), TestMode()]
            transcript = generate_transcript(RNG; mode=mode, num_stations=3)
            @test length(transcript.transmissions) > 0
            @test startswith(transcript.label, "<START>")
            @test endswith(transcript.label, "<END>")
            @test length(transcript.mode_name) > 0
            # Labels use [S1]..[S6], [TS]/[TE] for turn boundaries
            @test occursin(r"\[S[1-6]\]", transcript.label)
            @test occursin("[TS]", transcript.label)
            @test occursin("[TE]", transcript.label)
        end
    end

    @testset "Batch Transcript Generation" begin
        transcripts = generate_transcripts(RNG, 5)
        @test length(transcripts) == 5
        for t in transcripts
            @test length(t.transmissions) > 0
        end
    end

    @testset "Morse Code Table" begin
        @test length(char_to_morse('A')) == 2  # .-
        @test length(char_to_morse('S')) == 3  # ...
        @test length(char_to_morse('O')) == 3  # ---
        @test length(char_to_morse('0')) == 5

        @test is_prosign("AR")
        @test is_prosign("SK")
        @test !is_prosign("CQ")

        @test dot_units(Dot()) == 1
        @test dot_units(Dash()) == 3
        @test dot_units(WordGap()) == 7

        @test is_keyed(Dot())
        @test is_keyed(Dash())
        @test !is_keyed(SymbolGap())
        @test !is_keyed(CharGap())
    end

    @testset "Morse Timing" begin
        params = TimingParams(20.0)
        @test params.dot_duration ≈ 1.2 / 20.0

        events = text_to_timed_events(RNG, "CQ CQ", 25.0)
        @test length(events) > 0
        @test total_duration(events) > 0.0
    end

    @testset "Morse Encoding" begin
        scene = BandScene(RNG; num_stations=2)
        transcript = generate_transcript(RNG, scene)
        scene_events = encode_transcript(RNG, transcript, scene)

        @test length(scene_events.station_events) > 0
        @test scene_events.total_duration > 0.0
        @test length(scene_events.label) > 0
    end

    @testset "Envelope Generation" begin
        events = text_to_timed_events(RNG, "TEST", 20.0)
        dur = total_duration(events)

        env_rc = generate_envelope(events, dur, 44100;
            envelope_type=RaisedCosineEnvelope())
        @test length(env_rc) > 0
        @test maximum(env_rc) <= 1.0
        @test minimum(env_rc) >= 0.0

        env_hard = generate_envelope(events, dur, 44100;
            envelope_type=HardEnvelope())
        @test length(env_hard) > 0
    end

    @testset "CW Signal Generation" begin
        events = text_to_timed_events(RNG, "SOS", 20.0)
        dur = total_duration(events)
        env = generate_envelope(events, dur, 44100)
        sig = generate_cw_signal(env, 600.0, 44100)
        @test length(sig) == length(env)
        @test maximum(abs, sig) <= 1.0
    end

    @testset "Channel Effects" begin
        n = 44100
        sig = zeros(Float64, n)
        sig[1000:2000] .= 0.5

        noise = GaussianNoise{Float64}(0.01)
        sig_noisy = copy(sig)
        apply_noise!(RNG, sig_noisy, noise)
        @test sig_noisy != sig  # Noise was added

        sig_fade = ones(Float64, n)
        fading = SinusoidalFading{Float64}(0.5, 0.5, 0.0)
        apply_fading!(sig_fade, fading, 44100)
        @test minimum(sig_fade) < 1.0  # Fading reduced some samples
    end

    @testset "Signal Mixing" begin
        transcript = generate_transcript(RNG; num_stations=2)
        scene = BandScene(RNG; num_stations=2)
        scene_events = encode_transcript(RNG, transcript, scene)
        mixed = mix_signals(RNG, scene_events, scene)

        @test length(mixed.samples) > 0
        @test mixed.sample_rate == 44100
        @test mixed.duration > 0.0
        @test maximum(abs, mixed.samples) <= 1.0
    end

    @testset "Mel Filterbank" begin
        fb = MelFilterbank()
        @test fb.n_filters == 40
        @test size(fb.filters, 1) == 40

        @test hz_to_mel(0.0) ≈ 0.0
        @test mel_to_hz(0.0) ≈ 0.0
        @test mel_to_hz(hz_to_mel(440.0)) ≈ 440.0 atol=0.01

        # Apply to power spectrum
        pspec = ones(Float64, fb.fft_size ÷ 2 + 1)
        mel_e = apply_filterbank(fb, pspec)
        @test length(mel_e) == fb.n_filters
    end

    @testset "STFT" begin
        config = STFTConfig()
        sig = randn(RNG, Float64, 44100)
        stft_result = compute_stft(sig, config)
        @test size(stft_result, 1) == config.fft_size ÷ 2 + 1
        @test size(stft_result, 2) > 0

        pspec = power_spectrogram(stft_result)
        @test all(x -> x >= 0, pspec)

        lspec = log_power_spectrogram(stft_result)
        @test size(lspec) == size(stft_result)
    end

    @testset "Spectrogram Generation — Audio Path" begin
        transcript = generate_transcript(RNG; num_stations=2)
        scene = BandScene(RNG; num_stations=2)

        result, mixed, _ = generate_spectrogram(AudioPath(), RNG, transcript, scene)
        @test size(result.mel_spectrogram, 1) > 0
        @test size(result.mel_spectrogram, 2) > 0
        @test length(result.label) > 0
        @test length(mixed.samples) > 0
    end

    @testset "Spectrogram Generation — Direct Path" begin
        transcript = generate_transcript(RNG; num_stations=2)
        scene = BandScene(RNG; num_stations=2)

        result, _ = generate_spectrogram(DirectPath(), RNG, transcript, scene)
        @test size(result.mel_spectrogram, 1) > 0
        @test size(result.mel_spectrogram, 2) > 0
        @test length(result.label) > 0
    end

    @testset "Consistency Metrics" begin
        a = randn(RNG, Float64, 10, 20)
        b = a .+ 0.01 * randn(RNG, Float64, 10, 20)

        @test compare(L2SpectralError(), a, b) < 0.1
        @test compare(CosineSimilarity(), a, b) > 0.99
        @test compare(MeanAbsoluteError(), a, b) < 0.1

        # Self-comparison
        @test compare(L2SpectralError(), a, a) ≈ 0.0 atol=1e-10
        @test compare(CosineSimilarity(), a, a) ≈ 1.0 atol=1e-10
    end

    @testset "Dataset Generation" begin
        config = DatasetConfig(path=DirectPath(), stations=2:3)
        sample = generate_sample(RNG, config)
        @test size(sample.mel_spectrogram, 1) > 0
        @test length(sample.label) > 0
        @test haskey(sample.metadata, "mode")
        @test haskey(sample.metadata, "contest")
        @test (sample.token_timing isa NoTiming) || (length(sample.token_timing.token_start_frames) == length(sample.token_timing.token_end_frames))

        samples = generate_dataset(RNG, 3, config)
        @test length(samples) == 3
    end

    @testset "Error Simulation" begin
        text = "HELLO WORLD"
        # Run many times to ensure errors sometimes occur
        error_count = 0
        for _ in 1:100
            result = maybe_insert_error(RNG, text, BeginnerOp())
            result != text && (error_count += 1)
        end
        @test error_count > 0  # At least some errors occurred
    end

    @testset "Propagation Conditions" begin
        gp = good_propagation()
        mp = moderate_propagation()
        pp = poor_propagation()
        @test gp.path_loss_mean < mp.path_loss_mean < pp.path_loss_mean
        @test gp.qsb_probability < pp.qsb_probability
    end

end
