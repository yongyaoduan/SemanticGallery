import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest
import SemanticGalleryPersistence
@testable import SemanticGalleryML

final class ArtifactEncodingTests: XCTestCase {
    func testInstalledArtifactsEncodeARealFixtureImage() async throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        let paths = AppPaths(root: runtimeRoot)
        try FileManager.default.createDirectory(at: paths.artifactsRoot, withIntermediateDirectories: true)
        try copyTree(from: try artifactFixtureRoot(), to: paths.artifactsRoot)

        let imageURL = try XCTUnwrap(realFixtureImageURL())
        let service = Stage1SigLIP2EmbeddingService(paths: paths)

        let vector = try await service.encodeImage(at: imageURL)

        XCTAssertFalse(vector.isEmpty)
        XCTAssertGreaterThan(vector.reduce(0.0) { $0 + abs($1) }, 0)
    }

    func testSwiftPreprocessMatchesLegacyYellowCatPixels() async throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        let paths = AppPaths(root: runtimeRoot)
        try FileManager.default.createDirectory(at: paths.artifactsRoot, withIntermediateDirectories: true)
        try copyTree(from: try artifactFixtureRoot(), to: paths.artifactsRoot)

        let session = try await SigLIP2Support.loadSession(paths: paths)
        let imageURL = try semanticYellowCatFixtureRoot().appending(path: "study-01.jpg")
        let pixels = try SigLIP2Support.preprocessImage(
            at: imageURL,
            imageSize: session.config.visionConfig.imageSize,
            ciContext: CIContext()
        )

        XCTAssertEqual(pixels.shape.map { Int($0) }, [1, 224, 224, 3])
        let flattened = pixels.asArray(Float.self)
        let expectedPrefix: [Float] = [
            -0.9296875, -0.92578125, -0.9453125,
            -0.9296875, -0.92578125, -0.9453125,
            -0.9296875, -0.92578125, -0.9453125,
            -0.9296875, -0.92578125, -0.9453125,
        ]
        XCTAssertEqual(Array(flattened.prefix(expectedPrefix.count)).count, expectedPrefix.count)
        for (actual, expected) in zip(flattened.prefix(expectedPrefix.count), expectedPrefix) {
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }

        let mean = flattened.reduce(0.0, +) / Float(flattened.count)
        XCTAssertEqual(mean, -0.42149693, accuracy: 0.02)
    }

    func testLucasStage1RanksYellowCatImagesAheadOfDistractors() async throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        let paths = AppPaths(root: runtimeRoot)
        try FileManager.default.createDirectory(at: paths.artifactsRoot, withIntermediateDirectories: true)
        try copyTree(from: try artifactFixtureRoot(), to: paths.artifactsRoot)

        let folderURL = try semanticYellowCatFixtureRoot()
        let expectedTopResults = ["study-01.jpg", "study-02.jpg", "study-03.jpg"]
        let query = "a yellow tabby cat resting on the floor"
        let service = Stage1SigLIP2EmbeddingService(paths: paths)
        let queryVector = try await service.encodeText(query)

        var scored: [(String, Double)] = []
        for url in supportedImageURLs(in: folderURL) {
            let vector = try await service.encodeImage(at: url)
            let score = zip(queryVector, vector).reduce(0.0) { partial, pair in
                partial + pair.0 * pair.1
            }
            scored.append((url.lastPathComponent, score))
        }
        scored.sort { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0 < rhs.0
            }
            return lhs.1 > rhs.1
        }

        XCTAssertEqual(
            Set(scored.prefix(3).map(\.0)),
            Set(expectedTopResults),
            scored.map { "\($0.0): \(String(format: "%.6f", $0.1))" }.joined(separator: ", ")
        )
    }

    func testLucasStage1RanksTheReferenceCatImageAheadOfDistractors() async throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        let paths = AppPaths(root: runtimeRoot)
        try FileManager.default.createDirectory(at: paths.artifactsRoot, withIntermediateDirectories: true)
        try copyTree(from: try artifactFixtureRoot(), to: paths.artifactsRoot)

        let folderURL = try semanticYellowCatFixtureRoot()
        let service = Stage1SigLIP2EmbeddingService(paths: paths)
        let files = supportedImageURLs(in: folderURL)
        let referenceURL = try XCTUnwrap(files.first(where: { $0.lastPathComponent == "study-01.jpg" }))
        let queryVector = try await service.encodeImage(at: referenceURL)

        var scored: [(String, Double)] = []
        for url in files {
            let vector = try await service.encodeImage(at: url)
            let score = zip(queryVector, vector).reduce(0.0) { partial, pair in
                partial + pair.0 * pair.1
            }
            scored.append((url.lastPathComponent, score))
        }
        scored.sort { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0 < rhs.0
            }
            return lhs.1 > rhs.1
        }

        XCTAssertEqual(scored.first?.0, "study-01.jpg", scored.map { "\($0.0): \(String(format: "%.6f", $0.1))" }.joined(separator: ", "))
        XCTAssertTrue(
            Set(scored.prefix(3).map(\.0)).isSuperset(of: ["study-01.jpg", "study-02.jpg", "study-03.jpg"]),
            scored.map { "\($0.0): \(String(format: "%.6f", $0.1))" }.joined(separator: ", ")
        )
    }

    func testBFloat16ContrastiveTrainingStepRemainsFinite() {
        let adapter = ImageProjectionAdapter(dimension: 4, rank: 2, seed: 42)
        let lossAndGrad = valueAndGrad(model: adapter) { (adapter: ImageProjectionAdapter, arrays: [MLXArray]) in
            let adaptedImages = adapter(arrays[0])
            let loss = AdaptationTrainingMath.contrastiveLoss(
                imageEmbeddings: adaptedImages,
                textEmbeddings: arrays[1],
                logitScale: MLXArray(0.0, dtype: .bfloat16)
            )
            return [loss]
        }

        let images = MLXArray(
            [
                Float(0.7), Float(0.2), Float(0.1), Float(0.0),
                Float(0.2), Float(0.6), Float(0.2), Float(0.0),
            ],
            [2, 4]
        ).asType(.bfloat16)
        let texts = MLXArray(
            [
                Float(0.6), Float(0.3), Float(0.1), Float(0.0),
                Float(0.1), Float(0.7), Float(0.2), Float(0.0),
            ],
            [2, 4]
        ).asType(.bfloat16)

        let (values, gradients) = lossAndGrad(adapter, [images, texts])
        eval(values[0], gradients)

        XCTAssertTrue(values[0].item(Float.self).isFinite)
    }

    private func artifactFixtureRoot() throws -> URL {
        let rootPath = ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT"] ?? "/tmp/semanticgallery-ui-artifacts"
        let url = URL(filePath: rootPath, directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            throw NSError(domain: "ArtifactEncodingTests", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Artifact fixture root is missing."
            ])
        }
        return url
    }

    private func realFixtureImageURL() -> URL? {
        let fixtureRoot = ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT"]
            .map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? URL(filePath: "/tmp/semanticgallery-ui-fixtures", directoryHint: .isDirectory)
        let candidates = [
            fixtureRoot.appending(path: "SemanticGalleryUITestAlbumImageSearch"),
            URL(filePath: "/Users/duanyongyao/PythonProjects/phone_pictures", directoryHint: .isDirectory),
        ]

        for root in candidates where FileManager.default.fileExists(atPath: root.path(percentEncoded: false)) {
            if let url = supportedImageURLs(in: root).first {
                return url
            }
        }
        return nil
    }

    private func semanticYellowCatFixtureRoot() throws -> URL {
        let sources: [(filename: String, url: URL)] = [
            ("study-01.jpg", URL(string: "https://images.pexels.com/photos/9415244/pexels-photo-9415244.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
            ("study-02.jpg", URL(string: "https://images.pexels.com/photos/14440674/pexels-photo-14440674.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
            ("study-03.jpg", URL(string: "https://images.pexels.com/photos/208954/pexels-photo-208954.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
            ("study-04.jpg", URL(string: "https://images.pexels.com/photos/7543135/pexels-photo-7543135.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
            ("study-05.jpg", URL(string: "https://images.pexels.com/photos/11774609/pexels-photo-11774609.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
            ("study-06.jpg", URL(string: "https://images.pexels.com/photos/17078821/pexels-photo-17078821.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
            ("study-07.jpg", URL(string: "https://images.pexels.com/photos/14701951/pexels-photo-14701951.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500")!),
        ]
        let fixtureRoot = ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT"]
            .map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? FileManager.default.temporaryDirectory.appending(path: "semanticgallery-ui-fixtures")
        let expectedFiles = Set(sources.map(\.filename))
        let sharedURL = fixtureRoot.appending(path: "SemanticGalleryUITestSemanticYellowCat")
        if let existingFiles = try? FileManager.default.contentsOfDirectory(atPath: sharedURL.path),
           Set(existingFiles) == expectedFiles {
            return sharedURL
        }

        let url = FileManager.default.temporaryDirectory.appending(path: "semanticgallery-ui-yellow-cat-fixture")

        let fileManager = FileManager.default
        let existingFiles = try? fileManager.contentsOfDirectory(atPath: url.path)
        if Set(existingFiles ?? []) != expectedFiles {
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
            try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            for source in sources {
                let data = try Data(contentsOf: source.url)
                try data.write(to: url.appending(path: source.filename), options: .atomic)
            }
        }
        return url
    }

    private func supportedImageURLs(in root: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return enumerator.compactMap { candidate in
            guard let url = candidate as? URL else {
                return nil
            }
            let pathExtension = url.pathExtension.lowercased()
            return ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "gif", "bmp", "webp"].contains(pathExtension) ? url : nil
        }
        .sorted { $0.path(percentEncoded: false) < $1.path(percentEncoded: false) }
    }

    private func copyTree(from source: URL, to destination: URL) throws {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        for case let fileURL as URL in enumerator {
            let relativePath = relativePath(for: fileURL, inside: source)
            let destinationURL = destination.appending(path: relativePath)
            let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            if resourceValues.isRegularFile == true {
                try fileManager.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                if fileManager.fileExists(atPath: destinationURL.path) {
                    try fileManager.removeItem(at: destinationURL)
                }
                try fileManager.copyItem(at: fileURL, to: destinationURL)
            } else {
                try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
            }
        }
    }

    private func relativePath(for fileURL: URL, inside root: URL) -> String {
        let fileComponents = fileURL.standardizedFileURL.pathComponents
        let rootComponents = root.standardizedFileURL.pathComponents

        if fileComponents.starts(with: rootComponents) {
            return fileComponents.dropFirst(rootComponents.count).joined(separator: "/")
        }

        let privatePrefixedRootComponents: [String]
        if rootComponents.first == "/" {
            privatePrefixedRootComponents = ["/", "private"] + rootComponents.dropFirst()
        } else {
            privatePrefixedRootComponents = ["private"] + rootComponents
        }

        if fileComponents.starts(with: privatePrefixedRootComponents) {
            return fileComponents.dropFirst(privatePrefixedRootComponents.count).joined(separator: "/")
        }

        return fileURL.lastPathComponent
    }

    private func makeTemporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
