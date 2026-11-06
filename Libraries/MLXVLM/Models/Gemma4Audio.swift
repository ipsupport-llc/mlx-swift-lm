// Gemma 4 audio tower (E2B / E4B): a USM / Conformer-style encoder, its
// embedder into the text model, and the log-mel feature extractor.
//
// Ported from IPSupport's mlx-lm fork (`mlx_lm/models/gemma4_audio.py`),
// which mirrors transformers' `Gemma4AudioModel`. Weight keys match the
// checkpoint's `audio_tower.*` / `embed_audio.*` after the `model.` prefix
// is dropped; `sanitizeGemma4AudioWeights` handles the two differences.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4AudioConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let hiddenAct: String
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let residualWeight: Float
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let useClippedLinears: Bool
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let outputProjDims: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenAct = "hidden_act"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case residualWeight = "residual_weight"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case useClippedLinears = "use_clipped_linears"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case outputProjDims = "output_proj_dims"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        subsamplingConvChannels =
            try c.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels) ?? [128, 32]
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
        residualWeight = try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
        attentionChunkSize = try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
        attentionContextLeft =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
        attentionContextRight =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
        attentionLogitCap = try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50
        attentionInvalidLogitsValue =
            try c.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue) ?? -1e9
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? true
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        gradientClipping = try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
        outputProjDims = try c.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? 1536
    }
}

// MARK: - Layers

/// The largest finite value of a float dtype, as the reference clips to
/// `min(gradient_clipping, finfo(dtype).max)`.
private func gemma4AudioClipLimit(_ limit: Float, _ dtype: DType) -> Float {
    switch dtype {
    case .float16: return min(limit, 65504)
    default: return limit
    }
}

final class Gemma4AudioClippableLinear: Module, UnaryLayer {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(_ inFeatures: Int, _ outFeatures: Int, useClipping: Bool) {
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: false)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let inputMin, let inputMax { x = clip(x, min: inputMin, max: inputMax) }
        x = linear(x)
        if let outputMin, let outputMax { x = clip(x, min: outputMin, max: outputMax) }
        return x
    }
}

final class Gemma4AudioRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray?

    init(_ dimensions: Int, eps: Float, withScale: Bool = true) {
        self.eps = eps
        self._weight.wrappedValue = withScale ? MLXArray.ones([dimensions]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight ?? MLXArray.mlxNone, eps: eps)
    }
}

/// `[B, 1, numBlocks, chunk, context]` boolean mask: each query block sees
/// its sliding window of keys that are valid (not padding).
private func gemma4AudioLocalMask(
    _ valid: MLXArray, chunkSize: Int, maxPast: Int, maxFuture: Int
) -> MLXArray {
    let B = valid.dim(0)
    let L = valid.dim(1)
    let q = MLXArray(Int32(0) ..< Int32(L)).reshaped(-1, 1)
    let k = MLXArray(Int32(0) ..< Int32(L)).reshaped(1, -1)
    let dist = q - k
    let left = (dist .>= 0) .&& (dist .< maxPast)
    let right = (dist .< 0) .&& ((-dist) .< maxFuture)
    let window = left .|| right
    var mask = expandedDimensions(window, axis: 0) .&& expandedDimensions(valid, axis: 1)
    mask = expandedDimensions(mask, axis: 1)  // B, 1, L, L

    let numBlocks = (L + chunkSize - 1) / chunkSize
    let paddedL = numBlocks * chunkSize
    let padAmount = paddedL - L
    mask = padded(
        mask, widths: [0, 0, .init((0, padAmount)), .init((0, padAmount))],
        value: MLXArray(false))
    let contextSize = chunkSize + maxPast + maxFuture
    mask = mask.reshaped(B, 1, numBlocks, chunkSize, paddedL)
    mask = padded(
        mask, widths: [0, 0, 0, 0, .init((maxPast, maxFuture))], value: MLXArray(false))

    let blockStarts = MLXArray(Int32(0) ..< Int32(numBlocks)) * Int32(chunkSize)
    let offsets = MLXArray(Int32(0) ..< Int32(contextSize))
    var kvIndices = (blockStarts.reshaped(-1, 1) + offsets.reshaped(1, -1))
        .reshaped(1, 1, numBlocks, 1, contextSize)
    kvIndices = broadcast(kvIndices, to: [B, 1, numBlocks, chunkSize, contextSize])
    return takeAlong(mask, kvIndices, axis: -1)
}

final class Gemma4AudioAttention: Module {
    let numHeads: Int
    let headDim: Int
    let chunkSize: Int
    let maxPast: Int
    let maxFuture: Int
    let contextSize: Int
    let logitCap: Float
    let invalidLogit: Float
    let qScale: Float
    let kScale: Float

    @ModuleInfo(key: "q_proj") var qProj: Gemma4AudioClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4AudioClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4AudioClippableLinear
    @ModuleInfo(key: "post") var post: Gemma4AudioClippableLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray

    init(_ config: Gemma4AudioConfiguration) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        chunkSize = config.attentionChunkSize
        maxPast = config.attentionContextLeft - 1
        maxFuture = config.attentionContextRight
        contextSize = chunkSize + maxPast + maxFuture
        logitCap = config.attentionLogitCap
        invalidLogit = config.attentionInvalidLogitsValue
        qScale = Float(pow(Double(headDim), -0.5) / log(2.0))
        kScale = Float(log1p(M_E) / log(2.0))
        let h = config.hiddenSize
        let clip = config.useClippedLinears
        _qProj.wrappedValue = Gemma4AudioClippableLinear(h, numHeads * headDim, useClipping: clip)
        _kProj.wrappedValue = Gemma4AudioClippableLinear(h, numHeads * headDim, useClipping: clip)
        _vProj.wrappedValue = Gemma4AudioClippableLinear(h, numHeads * headDim, useClipping: clip)
        _post.wrappedValue = Gemma4AudioClippableLinear(h, h, useClipping: clip)
        _relativeKProj.wrappedValue = Linear(h, numHeads * headDim, bias: false)
        _perDimScale.wrappedValue = MLXArray.zeros([headDim])
        super.init()
    }

    /// `[B, L, H, D]` → non-overlapping `[B, numBlocks, chunk, H, D]`.
    private func toBlocks(_ x: MLXArray) -> MLXArray {
        let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let numBlocks = (L + chunkSize - 1) / chunkSize
        let pad = numBlocks * chunkSize - L
        let p = padded(x, widths: [0, .init((0, pad)), 0, 0])
        return p.reshaped(B, numBlocks, chunkSize, H, D)
    }

    /// Overlapping `context` windows per block: `[B, numBlocks, context, H, D]`.
    private func blockContext(_ x: MLXArray) -> MLXArray {
        let L = x.dim(1)
        let p = padded(x, widths: [0, .init((maxPast, maxFuture + chunkSize - 1)), 0, 0])
        let numBlocks = (L + chunkSize - 1) / chunkSize
        let starts = MLXArray(Int32(0) ..< Int32(numBlocks)) * Int32(chunkSize)
        let offsets = MLXArray(Int32(0) ..< Int32(contextSize))
        let indices = starts.reshaped(-1, 1) + offsets.reshaped(1, -1)
        return p.take(indices, axis: 1)
    }

    /// Transformer-XL relative shift for blocked attention.
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (B, H, nb, bs, pl) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var y = padded(x, widths: [0, 0, 0, 0, .init((0, contextSize + 1 - pl))])
        y = y.reshaped(B, H, nb, bs * (contextSize + 1))
        y = y[.ellipsis, ..<(bs * contextSize)]
        return y.reshaped(B, H, nb, bs, contextSize)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).asType(.float32).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).asType(.float32).reshaped(B, L, numHeads, headDim)
        var v = vProj(x).asType(.float32).reshaped(B, L, numHeads, headDim)

        q = q * qScale * softplus(perDimScale)
        k = k * kScale

        q = toBlocks(q)
        k = blockContext(k)
        v = blockContext(v)
        let numBlocks = q.dim(1)

        var relK = relativeKProj(positions).reshaped(-1, numHeads, headDim)
        relK = relK.asType(q.dtype)

        let queries = q.transposed(0, 3, 1, 2, 4)  // B, H, nb, chunk, D
        let matrixAC = matmul(queries, k.transposed(0, 3, 1, 4, 2))

        let queriesFlat = queries.reshaped(B, numHeads, -1, headDim)
        var matrixBD = matmul(queriesFlat, relK.transposed(1, 2, 0))
        matrixBD = matrixBD.reshaped(B, numHeads, numBlocks, chunkSize, -1)
        matrixBD = relShift(matrixBD)

        var weights = matrixAC + matrixBD
        weights = tanh(weights / logitCap) * logitCap
        weights = MLX.where(mask, weights, MLXArray(invalidLogit))
        weights = softmax(weights.asType(.float32), axis: -1).asType(v.dtype)

        var out = matmul(weights, v.transposed(0, 3, 1, 2, 4))  // B, H, nb, chunk, D
        out = out.transposed(0, 2, 3, 1, 4).reshaped(B, numBlocks * chunkSize, -1)
        out = out[0..., ..<L]
        return post(out.asType(x.dtype))
    }
}

final class Gemma4AudioSubSampleLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(inChannels: Int, outChannels: Int, eps: Float) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3,
            stride: 2, padding: 1, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: outChannels, eps: eps, affine: true, bias: false)
        super.init()
    }

    /// `x`: NHWC `[B, T, F, C]`; `mask`: `[B, T]` bool.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var h = x * expandedDimensions(mask, axes: [2, 3]).asType(x.dtype)
        h = conv(h.asType(conv.weight.dtype))
        h = relu(norm(h))
        return (h, mask[0..., .stride(by: 2)])
    }
}

final class Gemma4AudioSubSampleProjection: Module {
    @ModuleInfo(key: "layer0") var layer0: Gemma4AudioSubSampleLayer
    @ModuleInfo(key: "layer1") var layer1: Gemma4AudioSubSampleLayer
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(_ config: Gemma4AudioConfiguration) {
        let ch = config.subsamplingConvChannels
        _layer0.wrappedValue = Gemma4AudioSubSampleLayer(
            inChannels: 1, outChannels: ch[0], eps: config.rmsNormEps)
        _layer1.wrappedValue = Gemma4AudioSubSampleLayer(
            inChannels: ch[0], outChannels: ch[1], eps: config.rmsNormEps)
        _inputProjLinear.wrappedValue = Linear((ch[0] / 4) * ch[1], config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ features: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var (h, m) = layer0(expandedDimensions(features, axis: -1), mask: mask)
        (h, m) = layer1(h, mask: m)
        let (B, L) = (h.dim(0), h.dim(1))
        return (inputProjLinear(h.reshaped(B, L, -1)), m)
    }
}

final class Gemma4AudioFeedForward: Module {
    @ModuleInfo(key: "ffw_layer_1") var layer1: Gemma4AudioClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var layer2: Gemma4AudioClippableLinear
    @ModuleInfo(key: "pre_layer_norm") var preNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "post_layer_norm") var postNorm: Gemma4AudioRMSNorm
    let clipLimit: Float
    let postScale: Float

    init(_ config: Gemma4AudioConfiguration) {
        let h = config.hiddenSize
        _layer1.wrappedValue = Gemma4AudioClippableLinear(
            h, h * 4, useClipping: config.useClippedLinears)
        _layer2.wrappedValue = Gemma4AudioClippableLinear(
            h * 4, h, useClipping: config.useClippedLinears)
        _preNorm.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        _postNorm.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        clipLimit = config.gradientClipping
        postScale = config.residualWeight
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = gemma4AudioClipLimit(clipLimit, x.dtype)
        var h = clip(x, min: -c, max: c)
        h = preNorm(h)
        h = layer2(silu(layer1(h)))
        h = clip(h, min: -c, max: c)
        h = postNorm(h) * postScale
        return h + x
    }
}

final class Gemma4AudioCausalConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let leftPad: Int

    init(channels: Int, kernelSize: Int) {
        _conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            groups: channels, bias: false)
        leftPad = kernelSize - 1
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [0, .init((leftPad, 0)), 0]))
    }
}

final class Gemma4AudioLightConv1d: Module {
    @ModuleInfo(key: "linear_start") var linearStart: Gemma4AudioClippableLinear
    @ModuleInfo(key: "linear_end") var linearEnd: Gemma4AudioClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv: Gemma4AudioCausalConv1d
    @ModuleInfo(key: "pre_layer_norm") var preNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma4AudioRMSNorm
    let clipLimit: Float

    init(_ config: Gemma4AudioConfiguration) {
        let h = config.hiddenSize
        _linearStart.wrappedValue = Gemma4AudioClippableLinear(
            h, h * 2, useClipping: config.useClippedLinears)
        _linearEnd.wrappedValue = Gemma4AudioClippableLinear(
            h, h, useClipping: config.useClippedLinears)
        _depthwiseConv.wrappedValue = Gemma4AudioCausalConv1d(
            channels: h, kernelSize: config.convKernelSize)
        _preNorm.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        _convNorm.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        clipLimit = config.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = preNorm(x)
        h = glu(linearStart(h), axis: -1)
        h = depthwiseConv(h)
        let c = gemma4AudioClipLimit(clipLimit, h.dtype)
        h = clip(h, min: -c, max: c)
        h = convNorm(h)
        h = linearEnd(silu(h))
        return h + x
    }
}

final class Gemma4AudioEncoderLayer: Module {
    @ModuleInfo(key: "feed_forward1") var feedForward1: Gemma4AudioFeedForward
    @ModuleInfo(key: "feed_forward2") var feedForward2: Gemma4AudioFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4AudioAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: Gemma4AudioLightConv1d
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: Gemma4AudioRMSNorm
    let clipLimit: Float

    init(_ config: Gemma4AudioConfiguration) {
        let h = config.hiddenSize
        _feedForward1.wrappedValue = Gemma4AudioFeedForward(config)
        _feedForward2.wrappedValue = Gemma4AudioFeedForward(config)
        _selfAttn.wrappedValue = Gemma4AudioAttention(config)
        _lconv1d.wrappedValue = Gemma4AudioLightConv1d(config)
        _normPreAttn.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        _normPostAttn.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        _normOut.wrappedValue = Gemma4AudioRMSNorm(h, eps: config.rmsNormEps)
        clipLimit = config.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray) -> MLXArray {
        let c = gemma4AudioClipLimit(clipLimit, normPreAttn.weight?.dtype ?? x.dtype)
        var h = feedForward1(x)
        let residual = h
        h = normPreAttn(clip(h, min: -c, max: c))
        h = selfAttn(h, positions: positions, mask: mask)
        h = normPostAttn(clip(h, min: -c, max: c)) + residual
        h = lconv1d(h)
        h = feedForward2(h)
        return normOut(clip(h, min: -c, max: c))
    }
}

/// The audio encoder: log-mel features `[B, T, 128]` → soft tokens
/// `[B, ~T/4, outputProjDims]` and their validity mask.
public final class Gemma4AudioModel: Module {
    let config: Gemma4AudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") var subsample: Gemma4AudioSubSampleProjection
    @ModuleInfo(key: "layers") var layers: [Gemma4AudioEncoderLayer]
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    public init(_ config: Gemma4AudioConfiguration) {
        self.config = config
        _subsample.wrappedValue = Gemma4AudioSubSampleProjection(config)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Gemma4AudioEncoderLayer(config)
        }
        _outputProj.wrappedValue = Linear(config.hiddenSize, config.outputProjDims, bias: true)
        super.init()
    }

    /// Sinusoidal relative positions `[1, context/2 + 1, hidden]`, `[sin, cos]`.
    private func relativePositions(dtype: DType) -> MLXArray {
        let contextSize =
            config.attentionChunkSize + config.attentionContextLeft - 1
            + config.attentionContextRight
        let n = config.hiddenSize / 2
        let increment = log(10000.0) / Double(max(n - 1, 1))
        let inv = exp(MLXArray(0 ..< n).asType(.float32) * Float(-increment)).reshaped(1, 1, -1)
        let positions = MLXArray(
            stride(from: Int32(contextSize / 2), through: 0, by: -1).map { $0 }
        ).asType(.float32).reshaped(1, -1, 1)
        let t = positions * inv
        return concatenated([sin(t), cos(t)], axis: -1).asType(dtype)
    }

    public func callAsFunction(_ features: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var (h, valid) = subsample(features, mask: mask)
        let positions = relativePositions(dtype: h.dtype)
        let blockMask = gemma4AudioLocalMask(
            valid, chunkSize: config.attentionChunkSize,
            maxPast: config.attentionContextLeft - 1, maxFuture: config.attentionContextRight)
        for layer in layers {
            h = layer(h, mask: blockMask, positions: positions)
        }
        return (outputProj(h), valid)
    }
}

/// `embed_audio`: RMSNorm without scale, then a projection to the text width.
public final class Gemma4AudioEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var norm: Gemma4AudioRMSNorm

    public init(_ config: Gemma4AudioConfiguration, textHiddenSize: Int) {
        _embeddingProjection.wrappedValue = Linear(
            config.outputProjDims, textHiddenSize, bias: false)
        _norm.wrappedValue = Gemma4AudioRMSNorm(
            config.outputProjDims, eps: config.rmsNormEps, withScale: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(norm(x))
    }
}

/// Checkpoint keys → this module tree: `depthwise_conv1d.weight` lives under
/// `.conv`, and torch conv weights `(out, in, *k)` become mlx `(out, *k, in)`
/// (left alone when a converted checkpoint already has mlx's layout).
func sanitizeGemma4AudioWeights(
    _ weights: [String: MLXArray], expectedShapes: [String: [Int]]
) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (key, value) in weights {
        var k = key
        var v = value
        if k.hasSuffix("depthwise_conv1d.weight") {
            k = k.replacingOccurrences(of: "depthwise_conv1d.weight", with: "depthwise_conv1d.conv.weight")
        }
        if k.hasSuffix("depthwise_conv1d.conv.weight"), v.ndim == 3, expectedShapes[k] != v.shape {
            v = v.transposed(0, 2, 1)
        } else if k.hasSuffix("layer0.conv.weight") || k.hasSuffix("layer1.conv.weight"),
            v.ndim == 4, expectedShapes[k] != v.shape
        {
            v = v.transposed(0, 2, 3, 1)
        }
        out[k] = v
    }
    return out
}

// MARK: - Features

/// Log-mel features as transformers' `Gemma4AudioFeatureExtractor` computes
/// them (16 kHz mono; 20 ms Hann frames every 10 ms; |rfft 512|; 128 HTK mel
/// bins 0–8 kHz; log(mel + floor)). Computed on the CPU in float32.
public struct Gemma4AudioFeatureExtractor: Sendable {
    public let sampleRate: Int
    public let featureSize: Int
    public let frameLength: Int
    public let hopLength: Int
    public let fftLength: Int
    public let minFrequency: Double
    public let maxFrequency: Double
    public let melFloor: Float
    public let padToMultipleOf: Int
    public let inputScaleFactor: Float
    public let preemphasis: Float
    public let preemphasisHTKFlavor: Bool
    public let perBinMean: [Float]?
    public let perBinStddev: [Float]?
    /// Longest clip in samples (transformers' default `max_length`, 30 s).
    public let maxSamples: Int

    public init(
        sampleRate: Int = 16_000, featureSize: Int = 128, frameLengthMs: Double = 20,
        hopLengthMs: Double = 10, minFrequency: Double = 0, maxFrequency: Double = 8000,
        melFloor: Float = 1e-3, fftOverdrive: Bool = false, padToMultipleOf: Int = 128,
        inputScaleFactor: Float = 1, preemphasis: Float = 0, preemphasisHTKFlavor: Bool = true,
        perBinMean: [Float]? = nil, perBinStddev: [Float]? = nil, maxSamples: Int = 480_000
    ) {
        self.inputScaleFactor = inputScaleFactor
        self.preemphasis = preemphasis
        self.preemphasisHTKFlavor = preemphasisHTKFlavor
        self.perBinMean = perBinMean
        self.perBinStddev = perBinStddev
        self.maxSamples = maxSamples
        self.sampleRate = sampleRate
        self.featureSize = featureSize
        self.frameLength = Int((Double(sampleRate) * frameLengthMs / 1000).rounded())
        self.hopLength = Int((Double(sampleRate) * hopLengthMs / 1000).rounded())
        var fft = 1
        while fft < frameLength { fft *= 2 }
        self.fftLength = fftOverdrive ? fft * 2 : fft
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.melFloor = melFloor
        self.padToMultipleOf = padToMultipleOf
    }

    /// `[bins, mels]` triangular HTK filters, no normalization.
    func melFilters() -> [Float] {
        let bins = fftLength / 2 + 1
        func hzToMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let mMin = hzToMel(minFrequency)
        let mMax = hzToMel(maxFrequency)
        let edges = (0 ..< featureSize + 2).map {
            melToHz(mMin + (mMax - mMin) * Double($0) / Double(featureSize + 1))
        }
        var out = [Float](repeating: 0, count: bins * featureSize)
        for i in 0 ..< bins {
            let f = Double(sampleRate) / 2 * Double(i) / Double(bins - 1)
            for j in 0 ..< featureSize {
                let down = (f - edges[j]) / (edges[j + 1] - edges[j])
                let up = (edges[j + 2] - f) / (edges[j + 2] - edges[j + 1])
                out[i * featureSize + j] = Float(max(0, min(down, up)))
            }
        }
        return out
    }

    public enum ExtractionError: LocalizedError {
        case tooLong(seconds: Double, limit: Double)
        case tooShort
        public var errorDescription: String? {
            switch self {
            case .tooLong(let s, let l):
                String(format: "The audio clip is %.1f s; at most %.0f s is supported.", s, l)
            case .tooShort: "The audio clip is too short."
            }
        }
    }

    /// `samples`: 16 kHz mono. Returns features `[1, T, featureSize]` (zero
    /// where masked) and the frame mask `[1, T]`. Clips over `maxSamples` are
    /// refused, not cut (a cut clip would be answered as if whole).
    public func callAsFunction(_ samples: [Float]) throws -> (features: MLXArray, mask: MLXArray) {
        guard samples.count <= maxSamples else {
            throw ExtractionError.tooLong(
                seconds: Double(samples.count) / Double(sampleRate),
                limit: Double(maxSamples) / Double(sampleRate))
        }
        let multiple = padToMultipleOf
        let paddedCount = ((samples.count + multiple - 1) / multiple) * multiple
        let padLeft = frameLength / 2
        var wave = [Float](repeating: 0, count: padLeft + paddedCount)
        var valid = [Bool](repeating: false, count: padLeft + paddedCount)
        for (i, s) in samples.enumerated() {
            wave[padLeft + i] = s * inputScaleFactor
            valid[padLeft + i] = true
        }
        let unfold = frameLength + 1
        let frameCount = wave.count < unfold ? 0 : (wave.count - unfold) / hopLength + 1
        guard frameCount > 0 else { throw ExtractionError.tooShort }
        let bins = fftLength / 2 + 1

        // Each frame reads frameLength + 1 samples (preemphasis needs the
        // extra one); periodic Hann window; then a real DFT as two matmuls.
        let window = (0 ..< frameLength).map {
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(frameLength)))
        }
        let p = preemphasis
        var frames = [Float](repeating: 0, count: frameCount * frameLength)
        for t in 0 ..< frameCount {
            let x = wave[(t * hopLength) ..< (t * hopLength + unfold)].map { $0 }
            for n in 0 ..< frameLength {
                let v: Float
                if p <= 0 {
                    v = x[n]
                } else if preemphasisHTKFlavor {
                    v = n == 0 ? x[0] * (1 - p) : x[n] - p * x[n - 1]
                } else {
                    v = x[n + 1] - p * x[n]
                }
                frames[t * frameLength + n] = v * window[n]
            }
        }
        var cosM = [Float](repeating: 0, count: frameLength * bins)
        var sinM = [Float](repeating: 0, count: frameLength * bins)
        for n in 0 ..< frameLength {
            for k in 0 ..< bins {
                let a = 2 * Double.pi * Double(n * k % fftLength) / Double(fftLength)
                cosM[n * bins + k] = Float(cos(a))
                sinM[n * bins + k] = Float(-sin(a))
            }
        }
        var frameMask = [Bool](repeating: false, count: frameCount)
        for t in 0 ..< frameCount { frameMask[t] = valid[t * hopLength + unfold - 1] }

        return Stream.withNewDefaultStream(device: .cpu) {
            let f = MLXArray(frames, [frameCount, frameLength])
            let re = matmul(f, MLXArray(cosM, [frameLength, bins]))
            let im = matmul(f, MLXArray(sinM, [frameLength, bins]))
            let magnitude = sqrt(re * re + im * im)
            let mel = matmul(magnitude, MLXArray(melFilters(), [bins, featureSize]))
            let maskArray = MLXArray(frameMask.map { $0 ? Float(1) : 0 }, [frameCount, 1])
            var logMel = log(mel + melFloor)
            if let perBinMean { logMel = logMel - MLXArray(perBinMean, [1, featureSize]) }
            if let perBinStddev { logMel = logMel / MLXArray(perBinStddev, [1, featureSize]) }
            logMel = logMel * maskArray
            let features = expandedDimensions(logMel, axis: 0)
            let mask = MLXArray(frameMask, [1, frameCount])
            eval(features, mask)
            return (features, mask)
        }
    }

    /// Soft tokens the encoder makes from `frames` valid frames: two
    /// stride-2 subsamplings.
    public static func softTokenCount(frameMask: [Bool]) -> Int {
        var m = frameMask
        for _ in 0 ..< 2 { m = stride(from: 0, to: m.count, by: 2).map { m[$0] } }
        return m.filter { $0 }.count
    }
}
