import CoreImage
import CryptoKit
import Foundation
import MLX
import SemanticGalleryCore

extension PrivateAdaptationTrainer {
    func preparePublicExamples(
        examples: [PublicAnchorExample],
        session: SigLIPLoadedSession,
        batchSize: Int
    ) throws -> [PublicAnchorExample: PreparedPublicExample] {
        var preparedExamples: [PublicAnchorExample: PreparedPublicExample] = [:]

        for startIndex in stride(from: 0, to: examples.count, by: batchSize) {
            let endIndex = min(startIndex + batchSize, examples.count)
            let batch = Array(examples[startIndex..<endIndex])
            var pixelValues: [MLXArray] = []
            var inputIDs: [MLXArray] = []

            for example in batch {
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

            let imageFeatures = session.model.getImageFeatures(pixelValues: concatenated(pixelValues, axis: 0))
            let textFeatures = session.model.getTextFeatures(inputIds: concatenated(inputIDs, axis: 0), attentionMask: nil)
            eval(imageFeatures, textFeatures)

            let imageRows = featureRows(from: imageFeatures)
            let textRows = featureRows(from: textFeatures)
            for index in batch.indices {
                preparedExamples[batch[index]] = PreparedPublicExample(
                    imageFeatures: imageRows[index],
                    textFeatures: textRows[index]
                )
            }
        }

        return preparedExamples
    }

    func preparePrivateImages(
        images: [PrivateAdaptationImage],
        imageSize: Int,
        session: SigLIPLoadedSession,
        batchSize: Int
    ) throws -> [String: PreparedPrivateImage] {
        var preparedImagesByPath: [String: PreparedPrivateImage] = [:]

        for startIndex in stride(from: 0, to: images.count, by: batchSize) {
            let endIndex = min(startIndex + batchSize, images.count)
            let batch = Array(images[startIndex..<endIndex])
            var originals: [MLXArray] = []
            var viewA: [MLXArray] = []
            var viewB: [MLXArray] = []

            for image in batch {
                let original = try SigLIP2Support.preprocessImage(
                    at: URL(filePath: image.absolutePath),
                    imageSize: imageSize,
                    ciContext: ciContext
                )
                originals.append(original)
                viewA.append(jitter(original, brightnessShift: 0.04, contrast: 0.97))
                viewB.append(jitter(original, brightnessShift: -0.03, contrast: 1.03))
            }

            let originalFeatures = session.model.getImageFeatures(pixelValues: concatenated(originals, axis: 0))
            let viewAFeatures = session.model.getImageFeatures(pixelValues: concatenated(viewA, axis: 0))
            let viewBFeatures = session.model.getImageFeatures(pixelValues: concatenated(viewB, axis: 0))
            eval(originalFeatures, viewAFeatures, viewBFeatures)

            let originalRows = featureRows(from: originalFeatures)
            let viewARows = featureRows(from: viewAFeatures)
            let viewBRows = featureRows(from: viewBFeatures)
            for index in batch.indices {
                preparedImagesByPath[batch[index].absolutePath] = PreparedPrivateImage(
                    originalFeatures: originalRows[index],
                    viewAFeatures: viewARows[index],
                    viewBFeatures: viewBRows[index]
                )
            }
        }

        return preparedImagesByPath
    }

    func buildPublicBatch(
        examples: [PublicAnchorExample],
        preparedExamples: [PublicAnchorExample: PreparedPublicExample]
    ) throws -> PublicBatch {
        var imageFeatures: [MLXArray] = []
        var textFeatures: [MLXArray] = []

        for example in examples {
            guard let prepared = preparedExamples[example] else {
                throw PreparedTrainingDataError.missingPublicExample
            }
            imageFeatures.append(prepared.imageFeatures)
            textFeatures.append(prepared.textFeatures)
        }

        return PublicBatch(
            imageFeatures: concatenated(imageFeatures, axis: 0),
            textFeatures: concatenated(textFeatures, axis: 0)
        )
    }

    func buildPrivateBatch(
        images: [PrivateAdaptationImage],
        preparedImagesByPath: [String: PreparedPrivateImage]
    ) throws -> PrivateBatch {
        var originalFeatures: [MLXArray] = []
        var viewAFeatures: [MLXArray] = []
        var viewBFeatures: [MLXArray] = []

        for image in images {
            guard let prepared = preparedImagesByPath[image.absolutePath] else {
                throw PreparedTrainingDataError.missingPrivateImage
            }
            originalFeatures.append(prepared.originalFeatures)
            viewAFeatures.append(prepared.viewAFeatures)
            viewBFeatures.append(prepared.viewBFeatures)
        }

        return PrivateBatch(
            originalFeatures: concatenated(originalFeatures, axis: 0),
            viewAFeatures: concatenated(viewAFeatures, axis: 0),
            viewBFeatures: concatenated(viewBFeatures, axis: 0)
        )
    }

    func featureRows(from features: MLXArray) -> [MLXArray] {
        guard features.dim(0) > 1 else {
            return [features]
        }

        let featureDimension = features.shape[1]
        return (0..<features.dim(0)).map { index in
            features[index].reshaped(1, featureDimension)
        }
    }

    func jitter(_ array: MLXArray, brightnessShift: Float, contrast: Float) -> MLXArray {
        clip((array * MLXArray(contrast, dtype: array.dtype)) + MLXArray(brightnessShift, dtype: array.dtype), min: -1.0, max: 1.0)
    }

    func writePrivateManifest(_ manifest: [PrivateAdaptationImage], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let lines = try manifest.map { item in
            try String(
                decoding: JSONEncoder().encode(item),
                as: UTF8.self
            )
        }
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
    }

    func ensurePublicAnchorExtracted() throws -> URL {
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

    func loadPublicAnchorExamples(from root: URL) throws -> [PublicAnchorExample] {
        var examples = try loadFlickrExamples(from: root)
        examples.append(contentsOf: try loadScreen2WordsExamples(from: root))
        return examples
    }

    func loadFlickrExamples(from root: URL) throws -> [PublicAnchorExample] {
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

    func loadScreen2WordsExamples(from root: URL) throws -> [PublicAnchorExample] {
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

    func makePublicPool(
        from publicExamples: [PublicAnchorExample],
        limit: Int
    ) -> [PublicAnchorExample] {
        var generator = AdaptationSeededGenerator(state: 20_260_417)
        var shuffled = publicExamples
        shuffled.shuffle(using: &generator)
        return Array(shuffled.prefix(limit))
    }

    func sha256Prefix(for url: URL) -> String {
        let data = (try? Data(contentsOf: url)) ?? Data()
        return SHA256.hash(data: data).prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    func timedProgress(
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
            let remainingUnits = Swift.max(0, totalUnits - completedUnits)
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

    func splitCSVLine(_ line: String) -> [String] {
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
