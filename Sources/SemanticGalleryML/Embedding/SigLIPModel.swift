import Foundation
import MLX
import MLXNN

public struct SigLIPTextConfiguration: Decodable, Sendable {
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let layerNormEps: Float
    public let projectionSize: Int

    public init(
        vocabSize: Int = 256_000,
        maxPositionEmbeddings: Int = 64,
        hiddenSize: Int = 768,
        intermediateSize: Int = 3_072,
        numAttentionHeads: Int = 12,
        numHiddenLayers: Int = 12,
        layerNormEps: Float = 1e-6,
        projectionSize: Int = 768
    ) {
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numHiddenLayers = numHiddenLayers
        self.layerNormEps = layerNormEps
        self.projectionSize = projectionSize
    }

    private enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case layerNormEps = "layer_norm_eps"
        case projectionSize = "projection_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            vocabSize: try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 256_000,
            maxPositionEmbeddings: try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 64,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3_072,
            numAttentionHeads: try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12,
            numHiddenLayers: try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6,
            projectionSize: try container.decodeIfPresent(Int.self, forKey: .projectionSize) ?? 768
        )
    }
}

public struct SigLIPVisionConfiguration: Decodable, Sendable {
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let layerNormEps: Float
    public let numPatches: Int

    public init(
        imageSize: Int = 224,
        patchSize: Int = 16,
        numChannels: Int = 3,
        hiddenSize: Int = 768,
        intermediateSize: Int = 3_072,
        numAttentionHeads: Int = 12,
        numHiddenLayers: Int = 12,
        layerNormEps: Float = 1e-6,
        numPatches: Int = 196
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numHiddenLayers = numHiddenLayers
        self.layerNormEps = layerNormEps
        self.numPatches = numPatches
    }

    private enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case layerNormEps = "layer_norm_eps"
        case numPatches = "num_patches"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 224
        let patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        self.init(
            imageSize: imageSize,
            patchSize: patchSize,
            numChannels: try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3_072,
            numAttentionHeads: try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12,
            numHiddenLayers: try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6,
            numPatches: try container.decodeIfPresent(Int.self, forKey: .numPatches) ?? ((imageSize / patchSize) * (imageSize / patchSize))
        )
    }
}

public struct SigLIPConfiguration: Decodable, Sendable {
    public let textConfig: SigLIPTextConfiguration
    public let visionConfig: SigLIPVisionConfiguration

    public init(
        textConfig: SigLIPTextConfiguration = SigLIPTextConfiguration(),
        visionConfig: SigLIPVisionConfiguration = SigLIPVisionConfiguration()
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
    }

    private enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            textConfig: try container.decodeIfPresent(SigLIPTextConfiguration.self, forKey: .textConfig) ?? SigLIPTextConfiguration(),
            visionConfig: try container.decodeIfPresent(SigLIPVisionConfiguration.self, forKey: .visionConfig) ?? SigLIPVisionConfiguration()
        )
    }
}

private final class SigLIPAttention: Module {
    let numHeads: Int
    let normFactor: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(hiddenSize: Int, numHeads: Int, bias: Bool = true) {
        self.numHeads = numHeads
        self.normFactor = sqrt(Float(hiddenSize / numHeads))
        self._queryProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        self._keyProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        self._valueProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        self._outputProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var queries = queryProjection(x)
        var keys = keyProjection(x)
        var values = valueProjection(x)

        let (batchSize, queryLength) = (queries.dim(0), queries.dim(1))
        let keyLength = keys.dim(1)

        queries = queries.reshaped(batchSize, queryLength, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, keyLength, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, keyLength, numHeads, -1).transposed(0, 2, 1, 3)

        var scores = queries.matmul(keys.transposed(0, 1, 3, 2)) / normFactor
        if let mask {
            scores = scores + mask
        }
        let probabilities = softmax(scores, axis: -1)
        let attended = matmul(probabilities, values).transposed(0, 2, 1, 3).reshaped(batchSize, queryLength, -1)
        return outputProjection(attended)
    }
}

private final class SigLIPMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var firstProjection: Linear
    @ModuleInfo(key: "fc2") var secondProjection: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._firstProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._secondProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        secondProjection(geluApproximate(firstProjection(x)))
    }
}

private final class SigLIPEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SigLIPAttention
    @ModuleInfo(key: "layer_norm1") var preAttentionNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SigLIPMLP
    @ModuleInfo(key: "layer_norm2") var postAttentionNorm: LayerNorm

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int, layerNormEps: Float) {
        self._selfAttention.wrappedValue = SigLIPAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self._preAttentionNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)
        self._mlp.wrappedValue = SigLIPMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self._postAttentionNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attended = selfAttention(preAttentionNorm(x), mask: mask)
        let hidden = x + attended
        return hidden + mlp(postAttentionNorm(hidden))
    }
}

private final class SigLIPEncoder: Module {
    @ModuleInfo fileprivate var layers: [SigLIPEncoderLayer]

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int, numLayers: Int, layerNormEps: Float) {
        self.layers = (0 ..< numLayers).map { _ in
            SigLIPEncoderLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numHeads: numHeads,
                layerNormEps: layerNormEps
            )
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return hidden
    }
}

private final class SigLIPTextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(config: SigLIPTextConfiguration) {
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._positionEmbedding.wrappedValue = Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let sequenceLength = inputIds.dim(-1)
        let positionIds = broadcast(MLXArray.arange(sequenceLength), to: inputIds.shape)
        return tokenEmbedding(inputIds) + positionEmbedding(positionIds)
    }
}

private final class SigLIPTextTransformer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SigLIPTextEmbeddings
    @ModuleInfo(key: "encoder") var encoder: SigLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    @ModuleInfo(key: "head") var head: Linear

    init(config: SigLIPTextConfiguration) {
        self._embeddings.wrappedValue = SigLIPTextEmbeddings(config: config)
        self._encoder.wrappedValue = SigLIPEncoder(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            layerNormEps: config.layerNormEps
        )
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._head.wrappedValue = Linear(config.hiddenSize, config.projectionSize, bias: true)
    }

    func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let hiddenStates = embeddings(inputIds)
        let encoded = encoder(hiddenStates, mask: Self.makeMask(attentionMask))
        let normalized = finalLayerNorm(encoded)
        let pooled = normalized[0..., normalized.dim(1) - 1, 0...]
        return head(pooled)
    }

    private static func makeMask(_ attentionMask: MLXArray?) -> MLXArray? {
        guard let attentionMask else {
            return nil
        }

        let expanded = attentionMask
            .reshaped(attentionMask.dim(0), 1, 1, attentionMask.dim(1))
            .asType(SigLIPPrecisionPolicy.deployment)
        return (MLXArray(1.0, dtype: SigLIPPrecisionPolicy.deployment) - expanded)
            * MLXArray(-1_000_000_000.0, dtype: SigLIPPrecisionPolicy.deployment)
    }
}

private final class SigLIPTextWrapper: Module {
    @ModuleInfo(key: "text_model") var textModel: SigLIPTextTransformer

    init(config: SigLIPTextConfiguration) {
        self._textModel.wrappedValue = SigLIPTextTransformer(config: config)
    }

    func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        textModel(inputIds: inputIds, attentionMask: attentionMask)
    }
}

private final class SigLIPPoolingAttention: Module {
    let numHeads: Int
    let normFactor: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.normFactor = sqrt(Float(hiddenSize / numHeads))
        self._queryProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._keyProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._valueProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outputProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        var projectedQueries = queryProjection(queries)
        var projectedKeys = keyProjection(keys)
        var projectedValues = valueProjection(values)

        let batchSize = projectedQueries.dim(0)
        let queryLength = projectedQueries.dim(1)
        let keyLength = projectedKeys.dim(1)

        projectedQueries = projectedQueries.reshaped(batchSize, queryLength, numHeads, -1).transposed(0, 2, 1, 3)
        projectedKeys = projectedKeys.reshaped(batchSize, keyLength, numHeads, -1).transposed(0, 2, 1, 3)
        projectedValues = projectedValues.reshaped(batchSize, keyLength, numHeads, -1).transposed(0, 2, 1, 3)

        let scores = projectedQueries.matmul(projectedKeys.transposed(0, 1, 3, 2)) / normFactor
        let probabilities = softmax(scores, axis: -1)
        let attended = matmul(probabilities, projectedValues).transposed(0, 2, 1, 3).reshaped(batchSize, queryLength, -1)
        return outputProjection(attended)
    }
}

private final class SigLIPMultiheadAttentionPoolingHead: Module {
    let hiddenSize: Int

    @ParameterInfo(key: "probe") var probe: MLXArray
    @ModuleInfo(key: "attention") var attention: SigLIPPoolingAttention
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SigLIPMLP

    init(config: SigLIPVisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self._probe.wrappedValue = MLXArray.ones([1, 1, config.hiddenSize])
        self._attention.wrappedValue = SigLIPPoolingAttention(hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = SigLIPMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(_ hiddenState: MLXArray) -> MLXArray {
        let repeatedProbe = broadcast(probe, to: [hiddenState.dim(0), 1, hiddenSize])
        let attended = attention(repeatedProbe, keys: hiddenState, values: hiddenState)
        let normalized = layerNorm(attended)
        let pooled = attended + mlp(normalized)
        return pooled[0..., 0, 0...]
    }
}

private final class SigLIPVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let positions: Int

    init(config: SigLIPVisionConfiguration) {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize)
        )
        self.positions = config.numPatches
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: positions,
            dimensions: config.hiddenSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let patchEmbeddings = patchEmbedding(x).flattened(start: 1, end: 2)
        let positionIDs = MLXArray(0 ..< positions)[.newAxis, 0...]
        return patchEmbeddings + positionEmbedding(positionIDs)
    }
}

private final class SigLIPVisionTransformer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SigLIPVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: SigLIPEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
    @ModuleInfo(key: "head") var head: SigLIPMultiheadAttentionPoolingHead

    init(config: SigLIPVisionConfiguration) {
        self._embeddings.wrappedValue = SigLIPVisionEmbeddings(config: config)
        self._encoder.wrappedValue = SigLIPEncoder(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            layerNormEps: config.layerNormEps
        )
        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._head.wrappedValue = SigLIPMultiheadAttentionPoolingHead(config: config)
    }

    func callAsFunction(pixelValues: MLXArray) -> MLXArray {
        let embedded = embeddings(pixelValues)
        let encoded = encoder(embedded)
        return head(postLayerNorm(encoded))
    }
}

private final class SigLIPVisionWrapper: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SigLIPVisionTransformer

    init(config: SigLIPVisionConfiguration) {
        self._visionModel.wrappedValue = SigLIPVisionTransformer(config: config)
    }

    func callAsFunction(pixelValues: MLXArray) -> MLXArray {
        visionModel(pixelValues: pixelValues)
    }
}

public final class SigLIPModel: Module {
    let config: SigLIPConfiguration

    @ModuleInfo(key: "text_model") private var textTower: SigLIPTextWrapper
    @ModuleInfo(key: "vision_model") private var visionTower: SigLIPVisionWrapper
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray
    @ParameterInfo(key: "logit_bias") var logitBias: MLXArray

    public init(configuration: SigLIPConfiguration) {
        self.config = configuration
        self._textTower.wrappedValue = SigLIPTextWrapper(config: configuration.textConfig)
        self._visionTower.wrappedValue = SigLIPVisionWrapper(config: configuration.visionConfig)
        self._logitScale.wrappedValue = MLXArray(0.0, dtype: SigLIPPrecisionPolicy.deployment)
        self._logitBias.wrappedValue = MLXArray(0.0, dtype: SigLIPPrecisionPolicy.deployment)
    }

    public func getTextFeatures(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        textTower(inputIds: inputIds, attentionMask: attentionMask)
    }

    public func getImageFeatures(pixelValues: MLXArray) -> MLXArray {
        visionTower(pixelValues: pixelValues)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue
            }

            let normalizedKey = Self.normalizedWeightKey(key)

            if normalizedKey == "vision_model.vision_model.head.attention.in_proj.weight" {
                let splitWeights = split(
                    value,
                    indices: [
                        config.visionConfig.hiddenSize,
                        config.visionConfig.hiddenSize * 2,
                    ],
                    axis: 0
                )
                sanitized["vision_model.vision_model.head.attention.q_proj.weight"] = splitWeights[0]
                sanitized["vision_model.vision_model.head.attention.k_proj.weight"] = splitWeights[1]
                sanitized["vision_model.vision_model.head.attention.v_proj.weight"] = splitWeights[2]
                continue
            }

            if normalizedKey == "vision_model.vision_model.head.attention.in_proj.bias" {
                let splitBias = split(
                    value,
                    indices: [
                        config.visionConfig.hiddenSize,
                        config.visionConfig.hiddenSize * 2,
                    ],
                    axis: 0
                )
                sanitized["vision_model.vision_model.head.attention.q_proj.bias"] = splitBias[0]
                sanitized["vision_model.vision_model.head.attention.k_proj.bias"] = splitBias[1]
                sanitized["vision_model.vision_model.head.attention.v_proj.bias"] = splitBias[2]
                continue
            }

            if normalizedKey.contains("patch_embedding.weight"), value.ndim == 4, Self.isMLXConvolutionWeight(value) == false {
                sanitized[normalizedKey] = value.transposed(0, 2, 3, 1)
                continue
            }

            if ["logit_scale", "logit_bias"].contains(normalizedKey),
               value.ndim == 1,
               value.dim(0) == 1 {
                sanitized[normalizedKey] = value.squeezed()
                continue
            }

            sanitized[normalizedKey] = value
        }

        return sanitized
    }

    static func normalizedWeightKey(_ key: String) -> String {
        if key.hasPrefix("text_model."),
           key.hasPrefix("text_model.text_model.") == false {
            return "text_model.text_model." + key.dropFirst("text_model.".count)
        }

        if key.hasPrefix("vision_model."),
           key.hasPrefix("vision_model.vision_model.") == false {
            return "vision_model.vision_model." + key.dropFirst("vision_model.".count)
        }

        return key
    }

    private static func isMLXConvolutionWeight(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else {
            return false
        }
        let outChannels = array.dim(0)
        let kernelHeight = array.dim(1)
        let kernelWidth = array.dim(2)
        return outChannels >= kernelHeight && outChannels >= kernelWidth && kernelHeight == kernelWidth
    }
}
