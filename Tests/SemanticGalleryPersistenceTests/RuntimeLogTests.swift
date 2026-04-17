import Foundation
import Testing
@testable import SemanticGalleryPersistence

@Test
func runtimeLogAppendsUserFacingEntriesToTheConfiguredLogFile() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root, bundledArtifactsRoot: nil)

    SemanticGalleryRuntimeLog.record(
        "Preparing the search model from the app bundle.",
        level: .info,
        category: "model",
        paths: paths,
        metadata: [
            "artifacts_root": paths.runtimeArtifactsRoot.path(percentEncoded: false),
        ]
    )
    SemanticGalleryRuntimeLog.record(
        "Semantic search model files could not be loaded from the app bundle.",
        level: .error,
        category: "model",
        paths: paths
    )

    let logText = try String(contentsOf: paths.logsRoot.appending(path: "semanticgallery.log"), encoding: .utf8)
    #expect(logText.contains("Preparing the search model from the app bundle."))
    #expect(logText.contains("Semantic search model files could not be loaded from the app bundle."))
    #expect(logText.contains("artifacts_root="))
}
