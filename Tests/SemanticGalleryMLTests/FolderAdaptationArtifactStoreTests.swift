import Foundation
import Testing
@testable import SemanticGalleryML
@testable import SemanticGalleryPersistence

@Test
func folderAdaptationArtifactStorePersistsMetadataAndResolvesFolderSpecificEncoder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let store = FolderAdaptationArtifactStore(paths: paths)
    let folderURL = URL(filePath: "/tmp/Library")
    let folderKey = FolderAdaptationArtifactStore.folderKey(for: folderURL)

    let artifact = FolderAdaptationArtifact(
        folderKey: folderKey,
        folderPath: folderURL.path(percentEncoded: false),
        encoderVersion: "stage2-library-abcdef",
        adapterWeightsURL: paths.modelsRoot
            .appending(path: "Adapted")
            .appending(path: folderKey)
            .appending(path: "weights.safetensors"),
        summaryURL: paths.modelsRoot
            .appending(path: "Adapted")
            .appending(path: folderKey)
            .appending(path: "summary.json"),
        trainedImageCount: 100
    )

    try FileManager.default.createDirectory(at: artifact.adapterWeightsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("adapter".utf8).write(to: artifact.adapterWeightsURL)
    try store.save(artifact)

    let restored = try #require(store.artifact(for: folderURL))
    #expect(restored.folderKey == artifact.folderKey)
    #expect(restored.encoderVersion == artifact.encoderVersion)
    #expect(restored.trainedImageCount == 100)
}
