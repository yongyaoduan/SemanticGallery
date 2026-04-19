import CoreImage
import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import SemanticGalleryCore
import SemanticGalleryPersistence

enum AdaptationTrainingMath {
    static let accumulation: DType = .float32

    static func contrastiveLoss(imageEmbeddings: MLXArray, textEmbeddings: MLXArray, logitScale: MLXArray) -> MLXArray {
        let images = batchedMatrix(from: imageEmbeddings)
        let texts = batchedMatrix(from: textEmbeddings)
        let scale = exp(clip(logitScale.asType(accumulation), min: Float32(-6), max: Float32(6)))
        let logits = matmul(images, texts.transposed(1, 0)) * scale
        let labels = MLXArray((0..<logits.shape[0]).map(Int32.init), [logits.shape[0]])
        return MLXArray(Float32(0.5), dtype: accumulation) * (
            crossEntropy(logits: logits, targets: labels, reduction: .mean)
                + crossEntropy(logits: logits.transposed(1, 0), targets: labels, reduction: .mean)
        )
    }

    static func pairedImageLoss(firstEmbeddings: MLXArray, secondEmbeddings: MLXArray, logitScale: MLXArray) -> MLXArray {
        contrastiveLoss(imageEmbeddings: firstEmbeddings, textEmbeddings: secondEmbeddings, logitScale: logitScale)
    }

    static func distillationLoss(studentEmbeddings: MLXArray, teacherEmbeddings: MLXArray) -> MLXArray {
        let student = batchedMatrix(from: studentEmbeddings)
        let teacher = batchedMatrix(from: teacherEmbeddings)
        return mean(MLXArray(Float32(1), dtype: accumulation) - (student * teacher).sum(axis: -1))
    }

    static func batchedMatrix(from array: MLXArray) -> MLXArray {
        let normalized = SigLIP2Support.normalizeEmbeddings(array.asType(SigLIPPrecisionPolicy.training))
        let matrix: MLXArray
        switch normalized.ndim {
        case 0:
            matrix = normalized.reshaped(1, 1)
        case 1:
            matrix = normalized.reshaped(1, normalized.shape[0])
        case 2:
            matrix = normalized
        default:
            matrix = normalized.reshaped(normalized.dim(0), -1)
        }
        return matrix.asType(accumulation)
    }
}

public enum PrivateAdaptationError: Error, LocalizedError {
    case insufficientImages(Int)
    case missingPublicAnchor
    case unreadablePublicAnchor
    case emptyPublicAnchor

    public var errorDescription: String? {
        switch self {
        case .insufficientImages(let count):
            return "This folder currently has \(count) supported images. SemanticGallery needs at least 100 images before local adaptation is worth running."
        case .missingPublicAnchor:
            return "The public adaptation anchor is missing."
        case .unreadablePublicAnchor:
            return "The public adaptation anchor could not be prepared."
        case .emptyPublicAnchor:
            return "The public adaptation anchor did not contain usable examples."
        }
    }
}

public actor PrivateAdaptationTrainer: GalleryAdaptationTraining {
    let paths: AppPaths
    let artifactStore: FolderAdaptationArtifactStore
    let manifestBuilder: PrivateAdaptationManifestBuilder
    let ciContext = CIContext()

    public init(
        paths: AppPaths,
        artifactStore: FolderAdaptationArtifactStore? = nil,
        manifestBuilder: PrivateAdaptationManifestBuilder = .init()
    ) {
        self.paths = paths
        self.artifactStore = artifactStore ?? FolderAdaptationArtifactStore(paths: paths)
        self.manifestBuilder = manifestBuilder
    }

    public func existingArtifact(for folderURL: URL) async throws -> GalleryAdaptationResult? {
        guard let artifact = artifactStore.artifact(for: folderURL) else {
            return nil
        }
        return GalleryAdaptationResult(
            artifact: artifact,
            embeddingService: Stage1SigLIP2EmbeddingService(paths: paths, adapterArtifact: artifact)
        )
    }

    public func runAdaptation(
        for folderURL: URL,
        progress: @escaping @Sendable (PrivateAdaptationProgress) async -> Void
    ) async throws -> GalleryAdaptationResult {
        let privateManifest = try manifestBuilder.makeManifest(from: folderURL)
        guard privateManifest.count >= 100 else {
            throw PrivateAdaptationError.insufficientImages(privateManifest.count)
        }

        let folderKey = FolderAdaptationArtifactStore.folderKey(for: folderURL)
        let trainingDirectory = paths.datasetsRoot
            .appending(path: "PrivateAdaptation")
            .appending(path: folderKey)
        let privateManifestURL = trainingDirectory.appending(path: "private_manifest.jsonl")
        let summaryURL = artifactStore.baseDirectory(for: folderURL).appending(path: "summary.json")
        let weightsURL = artifactStore.baseDirectory(for: folderURL).appending(path: "weights.safetensors")

        let prepareStart = Date()
        try writePrivateManifest(privateManifest, to: privateManifestURL)
        await progress(
            timedProgress(
                step: .prepareData,
                message: "Selected \(privateManifest.count) local training images",
                stepProgress: 0.34,
                overallProgress: 0.11,
                startedAt: prepareStart
            )
        )

        let publicAnchorRoot = try ensurePublicAnchorExtracted()
        let publicExamples = try loadPublicAnchorExamples(from: publicAnchorRoot)
        guard publicExamples.isEmpty == false else {
            throw PrivateAdaptationError.emptyPublicAnchor
        }
        let defaults = PrivateAdaptationTrainingDefaults()
        let publicPool = makePublicPool(from: publicExamples, limit: defaults.publicPoolSize)
        let privateTrainingSet = Array(privateManifest.prefix(defaults.publicItemsPerEpoch))
        await progress(
            timedProgress(
                step: .prepareData,
                message: "Loaded \(publicPool.count) public anchor examples",
                stepProgress: 0.67,
                overallProgress: 0.22,
                startedAt: prepareStart
            )
        )

        let session = try await SigLIP2Support.loadSession(paths: paths)
        let preparedTrainingData = try prepareTrainingData(
            session: session,
            privateImages: privateTrainingSet,
            publicExamples: publicPool,
            batchSize: defaults.featureBatchSize
        )
        await progress(
            timedProgress(
                step: .prepareData,
                message: "Prepared fixed training features for \(privateTrainingSet.count) local images",
                stepProgress: 1.0,
                overallProgress: 1.0 / 3.0,
                startedAt: prepareStart,
                remainingSeconds: 0
            )
        )

        let trainStart = Date()
        let training = try await trainAdapter(
            privateImages: privateTrainingSet,
            publicExamples: publicPool,
            preparedTrainingData: preparedTrainingData,
            logitScale: session.model.logitScale,
            defaults: defaults
        ) { currentStep, totalSteps, lossValue in
            await progress(
                self.timedProgress(
                    step: .trainModel,
                    message: "Running step \(currentStep) of \(totalSteps) · loss \(String(format: "%.4f", lossValue))",
                    stepProgress: Double(currentStep) / Double(totalSteps),
                    overallProgress: (1.0 + (Double(currentStep) / Double(totalSteps))) / 3.0,
                    startedAt: trainStart,
                    totalUnits: totalSteps,
                    completedUnits: currentStep
                )
            )
        }

        try FileManager.default.createDirectory(at: weightsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try save(arrays: training.adapter.arraysForSaving(), metadata: [:], url: weightsURL)
        let encoderVersion = "stage2-\(folderKey)-\(sha256Prefix(for: weightsURL))"
        let summary = AdaptationSummary(
            encoderVersion: encoderVersion,
            folderPath: folderURL.path(percentEncoded: false),
            folderKey: folderKey,
            privateImageCount: privateManifest.count,
            publicExampleCount: publicPool.count,
            privateBatchSize: defaults.miniBatchSize,
            totalSteps: defaults.totalSteps,
            losses: training.lossHistory,
            precision: "bfloat16",
            trainedAt: ISO8601DateFormatter().string(from: Date())
        )
        let summaryData = try JSONEncoder().encode(summary)
        try summaryData.write(to: summaryURL, options: .atomic)

        let artifact = FolderAdaptationArtifact(
            folderKey: folderKey,
            folderPath: folderURL.path(percentEncoded: false),
            encoderVersion: encoderVersion,
            adapterWeightsURL: weightsURL,
            summaryURL: summaryURL,
            trainedImageCount: privateManifest.count
        )
        try artifactStore.save(artifact)

        return GalleryAdaptationResult(
            artifact: artifact,
            embeddingService: Stage1SigLIP2EmbeddingService(paths: paths, adapterArtifact: artifact)
        )
    }
}
