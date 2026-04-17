import Foundation
import Testing
@testable import SemanticGalleryML
@testable import SemanticGalleryPersistence

@Test
func stage1ArtifactsExposeExpectedTensorKeys() throws {
    let artifactRoot = try liveArtifactFixtureRoot()
    let modelDirectory = artifactRoot.appending(path: "mlx").appending(path: "siglip2-base-patch16-224-f32")
    let stage1Weights = artifactRoot.appending(path: "semanticgallery").appending(path: "stage1").appending(path: "weights.safetensors")

    let stage1Keys = try safetensorKeys(at: stage1Weights)

    #expect(FileManager.default.fileExists(atPath: modelDirectory.appending(path: "config.json").path))
    #expect(FileManager.default.fileExists(atPath: modelDirectory.appending(path: "tokenizer.json").path))
    #expect(stage1Keys.isEmpty == false)
    #expect(stage1Keys.count == 408)
    #expect(stage1Keys.contains("text_model.text_model.embeddings.token_embedding.weight"))
    #expect(stage1Keys.contains("vision_model.vision_model.head.attention.in_proj.weight"))
    #expect(stage1Keys.contains("vision_model.vision_model.head.mlp.fc1.weight"))
    #expect(stage1Keys.contains("text_model.text_model.head.weight"))
}

@Test
func stage1SessionLogsResolvedArtifactPathsWhenBundledFilesAreMissing() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let missingArtifactsRoot = root.appending(path: "MissingArtifacts")
    let paths = AppPaths(root: root, bundledArtifactsRoot: missingArtifactsRoot)

    do {
        _ = try await SigLIP2Support.loadSession(paths: paths)
        Issue.record("Expected the bundled model load to fail when artifacts are missing.")
    } catch let error as GalleryEmbeddingError {
        #expect(error == .invalidModelArtifact)
    }

    let logText = try String(
        contentsOf: paths.logsRoot.appending(path: "semanticgallery.log"),
        encoding: .utf8
    )
    #expect(logText.contains("Preparing the semantic search model from the app bundle."))
    #expect(logText.contains("Semantic search model files could not be loaded from the app bundle."))
    #expect(logText.contains(missingArtifactsRoot.path(percentEncoded: false)))
    #expect(logText.contains("missing_artifacts="))
    #expect(logText.contains("model_config"))
    #expect(logText.contains("tokenizer"))
}

private func liveArtifactFixtureRoot() throws -> URL {
    let candidates = [
        ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT"],
        "/tmp/semanticgallery-ui-artifacts",
        "/Users/\(NSUserName())/.semanticgallery-ui-artifacts",
    ]
        .compactMap { $0 }
        .map { URL(filePath: $0, directoryHint: .isDirectory) }

    guard let rootURL = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path(percentEncoded: false)) }) else {
        throw NSError(
            domain: "SemanticGalleryMLTests",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Artifact fixture root is missing."]
        )
    }
    return rootURL
}

private func safetensorKeys(at url: URL) throws -> Set<String> {
    let data = try Data(contentsOf: url)
    guard data.count >= 8 else {
        throw NSError(
            domain: "SemanticGalleryMLTests",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Safetensors header is incomplete."]
        )
    }

    let headerLength = data.prefix(8).enumerated().reduce(UInt64.zero) { partial, item in
        partial | (UInt64(item.element) << (UInt64(item.offset) * 8))
    }
    let headerStart = 8
    let headerEnd = headerStart + Int(headerLength)
    guard data.count >= headerEnd else {
        throw NSError(
            domain: "SemanticGalleryMLTests",
            code: 3,
            userInfo: [NSLocalizedDescriptionKey: "Safetensors header does not fit inside the file."]
        )
    }

    let headerData = data.subdata(in: headerStart..<headerEnd)
    let payload = try JSONSerialization.jsonObject(with: headerData)
    guard let dictionary = payload as? [String: Any] else {
        throw NSError(
            domain: "SemanticGalleryMLTests",
            code: 4,
            userInfo: [NSLocalizedDescriptionKey: "Safetensors header is not a dictionary."]
        )
    }

    return Set(dictionary.keys.filter { $0 != "__metadata__" })
}
