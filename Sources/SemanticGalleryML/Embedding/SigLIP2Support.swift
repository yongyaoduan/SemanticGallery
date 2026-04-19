import AppKit
import CoreImage
import Foundation
import MLX
import MLXNN
import SemanticGalleryPersistence
import Tokenizers

struct SigLIPLoadedSession {
    let model: SigLIPModel
    let tokenizer: any Tokenizer
    let config: SigLIPConfiguration
    let maxLength: Int
    let padTokenID: Int
}

enum SigLIP2Support {
    static func loadSession(paths: AppPaths) async throws -> SigLIPLoadedSession {
        let artifactsRoot = paths.runtimeArtifactsRoot
        let modelDirectory = artifactsRoot.appending(path: "mlx").appending(path: "siglip2-base-patch16-224-f32")
        let stage1Weights = artifactsRoot.appending(path: "semanticgallery").appending(path: "stage1").appending(path: "weights.safetensors")
        let baseMetadata = [
            "artifacts_root": artifactsRoot.path(percentEncoded: false),
            "bundle": Bundle.main.bundleURL.path(percentEncoded: false),
            "model_directory": modelDirectory.path(percentEncoded: false),
            "resource_root": Bundle.main.resourceURL?.path(percentEncoded: false) ?? "",
            "stage1_weights": stage1Weights.path(percentEncoded: false),
        ]

        SemanticGalleryRuntimeLog.record(
            "Preparing the semantic search model from the app bundle.",
            category: "model",
            paths: paths,
            metadata: baseMetadata
        )

        let requiredArtifacts = [
            ("model_config", modelDirectory.appending(path: "config.json")),
            ("tokenizer", modelDirectory.appending(path: "tokenizer.json")),
            ("tokenizer_config", modelDirectory.appending(path: "tokenizer_config.json")),
            ("special_tokens", modelDirectory.appending(path: "special_tokens_map.json")),
            ("preprocessor_config", modelDirectory.appending(path: "preprocessor_config.json")),
            ("stage1_weights", stage1Weights),
        ]
        let missingArtifacts = requiredArtifacts.filter { FileManager.default.fileExists(atPath: $0.1.path) == false }

        if missingArtifacts.isEmpty == false {
            let missingModelArtifacts = missingArtifacts.filter { $0.0 != "stage1_weights" }
            let missingNames = missingArtifacts.map(\.0).joined(separator: ",")
            let missingPaths = missingArtifacts.map { $0.1.path(percentEncoded: false) }.joined(separator: " | ")
            let message = missingModelArtifacts.isEmpty
                ? "Semantic search weights could not be loaded from the app bundle."
                : "Semantic search model files could not be loaded from the app bundle."
            SemanticGalleryRuntimeLog.record(
                message,
                level: .error,
                category: "model",
                paths: paths,
                metadata: baseMetadata.merging(
                    [
                        "missing_artifacts": missingNames,
                        "missing_paths": missingPaths,
                    ],
                    uniquingKeysWith: { _, new in new }
                )
            )
            if missingModelArtifacts.isEmpty {
                throw GalleryEmbeddingError.invalidStage1Weights
            }
            throw GalleryEmbeddingError.invalidModelArtifact
        }

        do {
            let configData = try Data(contentsOf: modelDirectory.appending(path: "config.json"))
            let config = try JSONDecoder().decode(SigLIPConfiguration.self, from: configData)
            let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
            let tokenizerConfig = try tokenizerConfig(from: modelDirectory)
            let padToken = tokenizerConfig["pad_token"] as? String
            let maxLength = tokenizerConfig["max_length"] as? Int ?? config.textConfig.maxPositionEmbeddings
            let padTokenID = padToken.flatMap { tokenizer.convertTokenToId($0) } ?? 0

            let model = try loadModel(
                config: config,
                modelDirectory: modelDirectory,
                stage1Weights: stage1Weights
            )

            SemanticGalleryRuntimeLog.record(
                "Semantic search model is ready.",
                category: "model",
                paths: paths,
                metadata: baseMetadata
            )

            return SigLIPLoadedSession(
                model: model,
                tokenizer: tokenizer,
                config: config,
                maxLength: maxLength,
                padTokenID: padTokenID
            )
        } catch {
            SemanticGalleryRuntimeLog.record(
                "Semantic search model could not finish loading.",
                level: .error,
                category: "model",
                paths: paths,
                metadata: baseMetadata.merging(
                    ["details": String(describing: error)],
                    uniquingKeysWith: { _, new in new }
                )
            )
            throw error
        }
    }

    static func preprocessImage(at url: URL, imageSize: Int, ciContext: CIContext) throws -> MLXArray {
        guard let nsImage = NSImage(contentsOf: url) else {
            throw GalleryEmbeddingError.invalidImageData
        }
        return try preprocess(nsImage: nsImage, imageSize: imageSize, ciContext: ciContext)
    }

    static func preprocessImage(data: Data, imageSize: Int, ciContext: CIContext) throws -> MLXArray {
        guard let nsImage = NSImage(data: data) else {
            throw GalleryEmbeddingError.invalidImageData
        }
        return try preprocess(nsImage: nsImage, imageSize: imageSize, ciContext: ciContext)
    }

    static func preprocess(nsImage: NSImage, imageSize: Int, ciContext: CIContext) throws -> MLXArray {
        var proposedRect = CGRect(origin: .zero, size: nsImage.size)
        guard let cgImage = nsImage.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) else {
            throw GalleryEmbeddingError.invalidImageData
        }

        let ciImage = CIImage(cgImage: cgImage)
        let targetSize = CGSize(width: imageSize, height: imageSize)
        let scaleTransform = CGAffineTransform(
            scaleX: targetSize.width / ciImage.extent.width,
            y: targetSize.height / ciImage.extent.height
        )
        let resized = ciImage.transformed(by: scaleTransform)
        let bounds = CGRect(origin: .zero, size: targetSize)

        let bytesPerPixel = 4 * MemoryLayout<Float32>.size
        let bytesPerRow = imageSize * bytesPerPixel
        var bitmap = Data(count: imageSize * imageSize * bytesPerPixel)
        bitmap.withUnsafeMutableBytes { buffer in
            ciContext.render(
                resized,
                toBitmap: buffer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: bounds,
                format: .RGBAf,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
            )
        }

        var array = MLXArray(bitmap, [imageSize, imageSize, 4], type: Float32.self)
        array = array[0..., 0..., ..<3]
        array = array.reshaped(1, imageSize, imageSize, 3)
        array = array.asType(SigLIPPrecisionPolicy.deployment)
        array = (array - MLXArray(0.5, dtype: SigLIPPrecisionPolicy.deployment)) / MLXArray(0.5, dtype: SigLIPPrecisionPolicy.deployment)
        return array
    }

    static func textInputIDs(
        for text: String,
        tokenizer: any Tokenizer,
        maxLength: Int,
        padTokenID: Int
    ) -> MLXArray {
        MLXArray(paddedTokenIDs(
            for: text,
            tokenizer: tokenizer,
            maxLength: maxLength,
            padTokenID: padTokenID
        ), [1, maxLength])
    }

    static func paddedTokenIDs(
        for text: String,
        tokenizer: any Tokenizer,
        maxLength: Int,
        padTokenID: Int
    ) -> [Int32] {
        let encoded = tokenizer.encode(text: text, addSpecialTokens: true)
        let inputLength = min(encoded.count, maxLength)
        let padded = Array(encoded.prefix(maxLength)) + Array(repeating: padTokenID, count: max(0, maxLength - inputLength))
        return padded.map(Int32.init)
    }

    static func normalized(vector: [Float]) -> [Double] {
        let magnitude = sqrt(vector.reduce(0.0) { partial, value in
            partial + Double(value * value)
        })
        guard magnitude > 0 else {
            return vector.map { _ in 0.0 }
        }
        return vector.map { Double($0) / magnitude }
    }

    static func normalizeEmbeddings(_ array: MLXArray) -> MLXArray {
        let squared = array * array
        let norms = sqrt(squared.sum(axis: -1, keepDims: true) + MLXArray(1e-12, dtype: array.dtype))
        return array / norms
    }

    private static func tokenizerConfig(from modelDirectory: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: modelDirectory.appending(path: "tokenizer_config.json"))
        let payload = try JSONSerialization.jsonObject(with: data)
        return payload as? [String: Any] ?? [:]
    }

    private static func loadModel(
        config: SigLIPConfiguration,
        modelDirectory: URL,
        stage1Weights: URL
    ) throws -> SigLIPModel {
        let model = SigLIPModel(configuration: config)
        let stage1Arrays = try loadArrays(url: stage1Weights)
        let sanitized = model.sanitize(weights: stage1Arrays)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        model.apply { array in
            array.dtype.isFloatingPoint ? array.asType(SigLIPPrecisionPolicy.deployment) : array
        }
        eval(model)
        return model
    }
}
