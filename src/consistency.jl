"""
    CWContestSim/src/spectrogram/consistency.jl

Consistency metrics for comparing mel-spectrograms generated
via the audio path (Mode 1) and direct path (Mode 2).
"""

# ============================================================================
# Metric Types
# ============================================================================

"""
    L2SpectralError <: AbstractConsistencyMetric

L2 (Euclidean) error between spectrograms.
"""
struct L2SpectralError <: AbstractConsistencyMetric end

"""
    CosineSimilarity <: AbstractConsistencyMetric

Cosine similarity between flattened spectrograms.
"""
struct CosineSimilarity <: AbstractConsistencyMetric end

"""
    KLDivergence <: AbstractConsistencyMetric

KL divergence between spectrograms (treated as distributions).
"""
struct KLDivergence <: AbstractConsistencyMetric end

"""
    MeanAbsoluteError <: AbstractConsistencyMetric

Mean absolute error between spectrograms.
"""
struct MeanAbsoluteError <: AbstractConsistencyMetric end

# ============================================================================
# Metric Computation via Dispatch
# ============================================================================

"""
    compare(metric, spec_a, spec_b) -> Float64

Compare two spectrograms using the given metric.
Lower values indicate better match for error metrics.
Higher values indicate better match for similarity metrics.
"""
function compare(::L2SpectralError,
                 a::Matrix{T}, b::Matrix{T}) where T<:AbstractFloat
    # Resize to common dimensions
    m = min(size(a, 1), size(b, 1))
    n = min(size(a, 2), size(b, 2))
    err = zero(T)
    @inbounds for j in 1:n
        for i in 1:m
            d = a[i, j] - b[i, j]
            err += d * d
        end
    end
    return sqrt(err / (m * n))
end

function compare(::CosineSimilarity,
                 a::Matrix{T}, b::Matrix{T}) where T<:AbstractFloat
    m = min(size(a, 1), size(b, 1))
    n = min(size(a, 2), size(b, 2))
    dot_ab = zero(T)
    norm_a = zero(T)
    norm_b = zero(T)
    @inbounds for j in 1:n
        for i in 1:m
            dot_ab += a[i, j] * b[i, j]
            norm_a += a[i, j]^2
            norm_b += b[i, j]^2
        end
    end
    denom = sqrt(norm_a) * sqrt(norm_b)
    denom < T(1e-15) && return zero(T)
    return dot_ab / denom
end

function compare(::KLDivergence,
                 a::Matrix{T}, b::Matrix{T}) where T<:AbstractFloat
    m = min(size(a, 1), size(b, 1))
    n = min(size(a, 2), size(b, 2))

    # Normalize to probability distributions (shift to positive, then normalize)
    a_min = minimum(view(a, 1:m, 1:n))
    b_min = minimum(view(b, 1:m, 1:n))
    offset = min(a_min, b_min) - one(T)

    sum_a = zero(T)
    sum_b = zero(T)
    @inbounds for j in 1:n, i in 1:m
        sum_a += a[i, j] - offset
        sum_b += b[i, j] - offset
    end

    kl = zero(T)
    eps = T(1e-15)
    @inbounds for j in 1:n
        for i in 1:m
            p = (a[i, j] - offset) / sum_a + eps
            q = (b[i, j] - offset) / sum_b + eps
            kl += p * log(p / q)
        end
    end
    return kl
end

function compare(::MeanAbsoluteError,
                 a::Matrix{T}, b::Matrix{T}) where T<:AbstractFloat
    m = min(size(a, 1), size(b, 1))
    n = min(size(a, 2), size(b, 2))
    err = zero(T)
    @inbounds for j in 1:n
        for i in 1:m
            err += abs(a[i, j] - b[i, j])
        end
    end
    return err / (m * n)
end

# ============================================================================
# Comprehensive Comparison Report
# ============================================================================

"""
    ConsistencyReport

Report comparing two spectrogram generation paths.
"""
struct ConsistencyReport
    l2_error::Float64
    cosine_similarity::Float64
    kl_divergence::Float64
    mae::Float64
end

"""
    compare_paths(spec_audio, spec_direct) -> ConsistencyReport

Run all consistency metrics between audio-path and direct-path spectrograms.
"""
function compare_paths(a::SpectrogramResult{T}, b::SpectrogramResult{T}) where T
    sa = a.mel_spectrogram
    sb = b.mel_spectrogram
    ConsistencyReport(
        compare(L2SpectralError(), sa, sb),
        compare(CosineSimilarity(), sa, sb),
        compare(KLDivergence(), sa, sb),
        compare(MeanAbsoluteError(), sa, sb)
    )
end

function Base.show(io::IO, r::ConsistencyReport)
    println(io, "Consistency Report:")
    println(io, "  L2 Spectral Error:  $(round(r.l2_error, digits=6))")
    println(io, "  Cosine Similarity:  $(round(r.cosine_similarity, digits=6))")
    println(io, "  KL Divergence:      $(round(r.kl_divergence, digits=6))")
    println(io, "  Mean Abs Error:     $(round(r.mae, digits=6))")
end
