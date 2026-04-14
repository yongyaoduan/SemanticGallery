import Foundation
import Testing
@testable import SemanticGalleryCore
@testable import SemanticGallerySettings

@Test
func folderPreparationUsesVisibleStepSequence() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let coordinator = FolderPreparationCoordinator()
    let steps = try await coordinator.prepareFolder(at: root).map(\.step)

    #expect(steps == [
        .requestFolderAccess,
        .scanSupportedImages,
        .persistBookmark,
        .finalizeFolderSelection,
    ])
}
