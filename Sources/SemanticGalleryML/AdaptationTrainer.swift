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
    private let paths: AppPaths
    private let artifactStore: FolderAdaptationArtifactStore
    private let manifestBuilder: PrivateAdaptationManifestBuilder
    private let ciContext = CIContext()

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
        await progress(
            timedProgress(
                step: .prepareData,
                message: "Loaded \(publicExamples.count) public anchor examples",
                stepProgress: 1.0,
                overallProgress: 1.0 / 3.0,
                startedAt: prepareStart,
                remainingSeconds: 0
            )
        )

        let session = try await SigLIP2Support.loadSession(paths: paths)
        let defaults = publicAnchorDefaults()
        let trainStart = Date()
        let training = try await trainAdapter(
            session: session,
            privateImages: privateManifest,
            publicExamples: publicExamples,
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
            publicExampleCount: publicExamples.count,
            privateBatchSize: defaults.privateBatchSize,
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

    private func trainAdapter(
        session: SigLIPLoadedSession,
        privateImages: [PrivateAdaptationImage],
        publicExamples: [PublicAnchorExample],
        defaults: PublicAnchorDefaults,
        onStep: @escaping @Sendable (Int, Int, Double) async -> Void
    ) async throws -> AdaptationTrainingResult {
        let adapter = ImageProjectionAdapter(dimension: session.config.textConfig.projectionSize, rank: 16)
        eval(adapter)
        let optimizer = AdamW(learningRate: 2e-4, weightDecay: 1e-2)
        let lossAndGrad = valueAndGrad(model: adapter) { (adapter: ImageProjectionAdapter, arrays: [MLXArray]) in
            let publicImageFeatures = session.model.getImageFeatures(pixelValues: arrays[0])
            let adaptedPublicImage = adapter(publicImageFeatures)
            let publicTextFeatures = session.model.getTextFeatures(inputIds: arrays[1], attentionMask: nil)
            let publicLoss = AdaptationTrainingMath.contrastiveLoss(
                imageEmbeddings: adaptedPublicImage,
                textEmbeddings: publicTextFeatures,
                logitScale: session.model.logitScale
            )

            let privateOriginalFeatures = session.model.getImageFeatures(pixelValues: arrays[2])
            let privateViewAFeatures = session.model.getImageFeatures(pixelValues: arrays[3])
            let privateViewBFeatures = session.model.getImageFeatures(pixelValues: arrays[4])
            let adaptedOriginal = adapter(privateOriginalFeatures)
            let adaptedViewA = adapter(privateViewAFeatures)
            let adaptedViewB = adapter(privateViewBFeatures)
            let privateLoss = AdaptationTrainingMath.pairedImageLoss(
                firstEmbeddings: adaptedViewA,
                secondEmbeddings: adaptedViewB,
                logitScale: session.model.logitScale
            )
            let distill = AdaptationTrainingMath.distillationLoss(
                studentEmbeddings: adaptedOriginal,
                teacherEmbeddings: privateOriginalFeatures
            )

            let total = publicLoss + (0.30 * privateLoss) + (0.15 * distill)
            return [total]
        }

        var privateCursor = 0
        var publicCursor = 0
        var lossHistory: [Double] = []
        for step in 1...defaults.totalSteps {
            let publicBatch = try buildPublicBatch(
                examples: publicExamples,
                cursor: &publicCursor,
                batchSize: defaults.publicBatchSize,
                session: session
            )
            let privateBatch = try buildPrivateBatch(
                images: privateImages,
                cursor: &privateCursor,
                batchSize: defaults.privateBatchSize,
                imageSize: session.config.visionConfig.imageSize
            )
            let arrays = [
                publicBatch.pixelValues,
                publicBatch.inputIDs,
                privateBatch.originals,
                privateBatch.viewA,
                privateBatch.viewB,
            ]

            let (values, gradients) = lossAndGrad(adapter, arrays)
            let loss = values[0]
            optimizer.update(model: adapter, gradients: gradients)
            eval(loss, adapter, optimizer)
            let lossValue = Double(loss.item(Float.self))
            lossHistory.append(lossValue)
            await onStep(step, defaults.totalSteps, lossValue)
        }

        return AdaptationTrainingResult(adapter: adapter, lossHistory: lossHistory)
    }

    private func buildPublicBatch(
        examples: [PublicAnchorExample],
        cursor: inout Int,
        batchSize: Int,
        session: SigLIPLoadedSession
    ) throws -> PublicBatch {
        var pixelValues: [MLXArray] = []
        var inputIDs: [MLXArray] = []

        for _ in 0..<batchSize {
            let example = examples[cursor % examples.count]
            cursor += 1
            let pixels = try SigLIP2Support.preprocessImage(
                at: example.imageURL,
                imageSize: session.config.visionConfig.imageSize,
                ciContext: ciContext
            )
            let ids = SigLIP2Support.textInputIDs(
                for: example.caption,
                tokenizer: session.tokenizer,
                maxLength: session.maxLength,
                padTokenID: session.padTokenID
            )
            pixelValues.append(pixels)
            inputIDs.append(ids)
        }

        return PublicBatch(
            pixelValues: concatenated(pixelValues, axis: 0),
            inputIDs: concatenated(inputIDs, axis: 0)
        )
    }

    private func buildPrivateBatch(
        images: [PrivateAdaptationImage],
        cursor: inout Int,
        batchSize: Int,
        imageSize: Int
    ) throws -> PrivateBatch {
        var originals: [MLXArray] = []
        var viewA: [MLXArray] = []
        var viewB: [MLXArray] = []

        for offset in 0..<batchSize {
            let image = images[(cursor + offset) % images.count]
            let original = try SigLIP2Support.preprocessImage(
                at: URL(filePath: image.absolutePath),
                imageSize: imageSize,
                ciContext: ciContext
            )
            originals.append(original)
            viewA.append(jitter(original, brightnessShift: 0.04, contrast: 0.97))
            viewB.append(jitter(original, brightnessShift: -0.03, contrast: 1.03))
        }
        cursor += batchSize

        return PrivateBatch(
            originals: concatenated(originals, axis: 0),
            viewA: concatenated(viewA, axis: 0),
            viewB: concatenated(viewB, axis: 0)
        )
    }

    private func jitter(_ array: MLXArray, brightnessShift: Float, contrast: Float) -> MLXArray {
        clip((array * MLXArray(contrast, dtype: array.dtype)) + MLXArray(brightnessShift, dtype: array.dtype), min: -1.0, max: 1.0)
    }

    private func writePrivateManifest(_ manifest: [PrivateAdaptationImage], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let lines = try manifest.map { item in
            try String(
                decoding: JSONEncoder().encode(item),
                as: UTF8.self
            )
        }
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: String.Encoding.utf8)
    }

    private func ensurePublicAnchorExtracted() throws -> URL {
        let archiveRoot = paths.runtimeArtifactsRoot.appending(path: "semanticgallery").appending(path: "stage2_public_anchor")
        let archiveURL = archiveRoot.appending(path: "semanticgallery-stage2-public-anchor.tar.gz")
        guard FileManager.default.fileExists(atPath: archiveURL.path(percentEncoded: false)) else {
            throw PrivateAdaptationError.missingPublicAnchor
        }

        let extractedRoot = archiveRoot.appending(path: "extracted")
        let sentinel = extractedRoot.appending(path: "flickr30k").appending(path: "captions.txt")
        if FileManager.default.fileExists(atPath: sentinel.path(percentEncoded: false)) {
            return extractedRoot
        }

        try FileManager.default.createDirectory(at: extractedRoot, withIntermediateDirectories: true)
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveURL.path(percentEncoded: false), "-C", extractedRoot.path(percentEncoded: false)]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw PrivateAdaptationError.unreadablePublicAnchor
        }
        guard FileManager.default.fileExists(atPath: sentinel.path(percentEncoded: false)) else {
            throw PrivateAdaptationError.unreadablePublicAnchor
        }
        return extractedRoot
    }

    private func loadPublicAnchorExamples(from root: URL) throws -> [PublicAnchorExample] {
        var examples = try loadFlickrExamples(from: root)
        examples.append(contentsOf: try loadScreen2WordsExamples(from: root))
        return examples
    }

    private func loadFlickrExamples(from root: URL) throws -> [PublicAnchorExample] {
        let captionsURL = root.appending(path: "flickr30k").appending(path: "captions.txt")
        guard let content = try? String(contentsOf: captionsURL, encoding: .utf8) else {
            return []
        }

        var captionsByImage: [String: [String]] = [:]
        for line in content.split(separator: "\n").dropFirst() {
            let columns = splitCSVLine(String(line))
            guard columns.count >= 3 else {
                continue
            }
            captionsByImage[columns[0], default: []].append(columns[2])
        }

        return captionsByImage.keys.sorted().compactMap { filename in
            let imageURL = root
                .appending(path: "flickr30k")
                .appending(path: "flickr30k_images")
                .appending(path: filename)
            guard FileManager.default.fileExists(atPath: imageURL.path(percentEncoded: false)) else {
                return nil
            }
            guard let caption = captionsByImage[filename]?.first else {
                return nil
            }
            return PublicAnchorExample(imageURL: imageURL, caption: caption)
        }
    }

    private func loadScreen2WordsExamples(from root: URL) throws -> [PublicAnchorExample] {
        let manifestURL = root.appending(path: "screen2words").appending(path: "manifest.jsonl")
        guard let content = try? String(contentsOf: manifestURL, encoding: .utf8) else {
            return []
        }

        return try content.split(separator: "\n").compactMap { line in
            let payload = try JSONSerialization.jsonObject(with: Data(line.utf8))
            guard let object = payload as? [String: Any] else {
                return nil
            }
            guard
                let rawPath = object["image_path"] as? String,
                let captions = object["captions"] as? [String],
                let caption = captions.first
            else {
                return nil
            }

            let pathComponents = URL(filePath: rawPath).pathComponents
            guard let screen2WordsIndex = pathComponents.firstIndex(of: "screen2words") else {
                return nil
            }
            let relativeComponents = pathComponents.dropFirst(screen2WordsIndex + 1)
            let imageURL = relativeComponents.reduce(root.appending(path: "screen2words")) { partial, component in
                partial.appending(path: component)
            }
            guard FileManager.default.fileExists(atPath: imageURL.path(percentEncoded: false)) else {
                return nil
            }
            return PublicAnchorExample(imageURL: imageURL, caption: caption)
        }
    }

    private func publicAnchorDefaults() -> PublicAnchorDefaults {
        let sampleInfoURL = paths.runtimeArtifactsRoot
            .appending(path: "semanticgallery")
            .appending(path: "stage2_public_anchor")
            .appending(path: "sample_info.json")
        guard
            let data = try? Data(contentsOf: sampleInfoURL),
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return PublicAnchorDefaults(privateBatchSize: 8, publicBatchSize: 4, totalSteps: 26)
        }
        let privateBatchSize = payload["private_batch_size"] as? Int ?? 8
        let totalSteps = payload["default_private_steps"] as? Int ?? 26
        return PublicAnchorDefaults(
            privateBatchSize: max(1, privateBatchSize),
            publicBatchSize: 4,
            totalSteps: max(1, totalSteps)
        )
    }

    private func sha256Prefix(for url: URL) -> String {
        let data = (try? Data(contentsOf: url)) ?? Data()
        return SHA256.hash(data: data).prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    private func timedProgress(
        step: PrivateAdaptationStep,
        message: String,
        stepProgress: Double,
        overallProgress: Double,
        startedAt: Date,
        totalUnits: Int? = nil,
        completedUnits: Int? = nil,
        remainingSeconds: Int? = nil
    ) -> PrivateAdaptationProgress {
        let elapsed = Int(Date().timeIntervalSince(startedAt))
        let resolvedRemaining: Int?
        if let remainingSeconds {
            resolvedRemaining = remainingSeconds
        } else if let totalUnits, let completedUnits, completedUnits > 0 {
            let remainingUnits = max(0, totalUnits - completedUnits)
            resolvedRemaining = Int((Double(elapsed) / Double(completedUnits)) * Double(remainingUnits))
        } else {
            resolvedRemaining = nil
        }

        return PrivateAdaptationProgress(
            step: step,
            message: message,
            stepProgress: stepProgress,
            overallProgress: overallProgress,
            elapsedSeconds: elapsed,
            remainingSeconds: resolvedRemaining
        )
    }

    private func splitCSVLine(_ line: String) -> [String] {
        var values: [String] = []
        var current = ""
        var insideQuotes = false

        for character in line {
            switch character {
            case "\"":
                insideQuotes.toggle()
            case "," where insideQuotes == false:
                values.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        values.append(current)
        return values
    }
}

private struct PublicAnchorExample: Sendable {
    let imageURL: URL
    let caption: String
}

private struct PublicBatch {
    let pixelValues: MLXArray
    let inputIDs: MLXArray
}

private struct PrivateBatch {
    let originals: MLXArray
    let viewA: MLXArray
    let viewB: MLXArray
}

private struct PublicAnchorDefaults {
    let privateBatchSize: Int
    let publicBatchSize: Int
    let totalSteps: Int
}

private struct AdaptationTrainingResult {
    let adapter: ImageProjectionAdapter
    let lossHistory: [Double]
}

private struct AdaptationSummary: Codable {
    let encoderVersion: String
    let folderPath: String
    let folderKey: String
    let privateImageCount: Int
    let publicExampleCount: Int
    let privateBatchSize: Int
    let totalSteps: Int
    let losses: [Double]
    let precision: String
    let trainedAt: String
}
