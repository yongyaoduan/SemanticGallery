import Foundation
import Testing
@testable import SemanticGalleryML
@testable import SemanticGalleryPersistence

@Test
func stage1EmbeddingServiceReturnsNoVectorsForAnEmptyBatch() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root, bundledArtifactsRoot: root.appending(path: "Artifacts"))
    let service = Stage1SigLIP2EmbeddingService(paths: paths)

    let embeddings = try await service.encodeImages(at: [])

    #expect(embeddings.isEmpty)
}
