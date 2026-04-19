import CoreImage
import Foundation
import MLX
import SemanticGalleryPersistence

public enum GalleryEmbeddingError: Error, Equatable {
    case invalidModelArtifact
    case invalidStage1Weights
    case invalidImageData
    case invalidAdapterArtifact
}

public actor Stage1SigLIP2EmbeddingService: GalleryEmbeddingService {
    public let encoderVersion: String

    private let paths: AppPaths
    private let adapterArtifact: FolderAdaptationArtifact?
    private let ciContext = CIContext()
    private var session: SigLIPLoadedSession?
    private var adapter: ImageProjectionAdapter?
    private var textQueryWarmupComplete = false
    private var imageQueryWarmupComplete = false

    public init(paths: AppPaths, adapterArtifact: FolderAdaptationArtifact? = nil) {
        self.paths = paths
        self.adapterArtifact = adapterArtifact
        self.encoderVersion = adapterArtifact?.encoderVersion ?? "stage1"
    }

    public func encodeImage(at url: URL) async throws -> [Double] {
        let loaded = try await loadedSession()
        let pixelValues = try SigLIP2Support.preprocessImage(at: url, imageSize: loaded.config.visionConfig.imageSize, ciContext: ciContext)
        let features = loaded.model.getImageFeatures(pixelValues: pixelValues)
        let adapted = try adaptedImageFeatures(from: features)
        eval(adapted)
        return SigLIP2Support.normalized(vector: adapted[0].asArray(Float.self))
    }

    public func encodeImages(at urls: [URL]) async throws -> [[Double]] {
        guard urls.isEmpty == false else {
            return []
        }

        let loaded = try await loadedSession()
        let pixelValues = try urls.map {
            try SigLIP2Support.preprocessImage(
                at: $0,
                imageSize: loaded.config.visionConfig.imageSize,
                ciContext: ciContext
            )
        }
        let batch = concatenated(pixelValues, axis: 0)
        let features = loaded.model.getImageFeatures(pixelValues: batch)
        let adapted = try adaptedImageFeatures(from: features)
        eval(adapted)

        return (0..<urls.count).map { index in
            SigLIP2Support.normalized(vector: adapted[index].asArray(Float.self))
        }
    }

    public func encodeImage(data: Data) async throws -> [Double] {
        let loaded = try await loadedSession()
        let pixelValues = try SigLIP2Support.preprocessImage(data: data, imageSize: loaded.config.visionConfig.imageSize, ciContext: ciContext)
        let features = loaded.model.getImageFeatures(pixelValues: pixelValues)
        let adapted = try adaptedImageFeatures(from: features)
        eval(adapted)
        return SigLIP2Support.normalized(vector: adapted[0].asArray(Float.self))
    }

    public func encodeText(_ text: String) async throws -> [Double] {
        let loaded = try await loadedSession()
        let inputIDs = SigLIP2Support.textInputIDs(
            for: text,
            tokenizer: loaded.tokenizer,
            maxLength: loaded.maxLength,
            padTokenID: loaded.padTokenID
        )
        let features = loaded.model.getTextFeatures(inputIds: inputIDs, attentionMask: nil)
        eval(features)
        return SigLIP2Support.normalized(vector: features[0].asArray(Float.self))
    }

    public func prepareForQueries() async throws {
        _ = try await loadedSession()
        if let adapterArtifact {
            _ = try loadAdapter(for: adapterArtifact)
        }
        if textQueryWarmupComplete == false {
            _ = try await encodeText("warmup")
            textQueryWarmupComplete = true
        }
        if imageQueryWarmupComplete == false {
            _ = try await encodeImage(data: Self.warmupPNGData)
            imageQueryWarmupComplete = true
        }
    }

    private func loadedSession() async throws -> SigLIPLoadedSession {
        if let session {
            return session
        }

        let session = try await SigLIP2Support.loadSession(paths: paths)
        self.session = session
        return session
    }

    private func adaptedImageFeatures(from features: MLXArray) throws -> MLXArray {
        guard let adapterArtifact else {
            return features
        }
        let adapter = try loadAdapter(for: adapterArtifact)
        return adapter(features.asType(SigLIPPrecisionPolicy.deployment))
    }

    private func loadAdapter(for artifact: FolderAdaptationArtifact) throws -> ImageProjectionAdapter {
        if let adapter {
            return adapter
        }
        let metadata = [
            "adapter_weights": artifact.adapterWeightsURL.path(percentEncoded: false),
            "encoder_version": artifact.encoderVersion,
            "folder": artifact.folderPath,
        ]
        SemanticGalleryRuntimeLog.record(
            "Loading the folder-specific semantic search adaptation.",
            category: "adaptation",
            paths: paths,
            metadata: metadata
        )
        guard FileManager.default.fileExists(atPath: artifact.adapterWeightsURL.path(percentEncoded: false)) else {
            SemanticGalleryRuntimeLog.record(
                "Semantic search could not load the folder adaptation, so only the published encoder is available.",
                level: .error,
                category: "adaptation",
                paths: paths,
                metadata: metadata
            )
            throw GalleryEmbeddingError.invalidAdapterArtifact
        }
        let arrays = try loadArrays(url: artifact.adapterWeightsURL)
        let adapter = try ImageProjectionAdapter(arrays: arrays)
        self.adapter = adapter
        SemanticGalleryRuntimeLog.record(
            "The folder-specific semantic search adaptation is ready.",
            category: "adaptation",
            paths: paths,
            metadata: metadata
        )
        return adapter
    }
}

private extension Stage1SigLIP2EmbeddingService {
    static let warmupPNGData =
        Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/8n8AAAAASUVORK5CYII=")
        ?? Data()
}
