import Foundation
import Testing
@testable import SemanticGalleryInstall
@testable import SemanticGalleryPersistence

@Test
func installCoordinatorEmitsLegacyStepSequence() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let coordinator = InstallCoordinator(
        paths: AppPaths(root: root),
        downloader: ArtifactDownloader.stubbed
    )

    let steps = try await coordinator.prepare().map(\.step)
    #expect(steps == [
        .prepareDirectories,
        .prepareDatabase,
        .downloadBaseModel,
        .downloadStage1Checkpoint,
        .downloadPublicAnchor,
        .verifyArtifacts,
        .finalizeInstallation,
    ])
}

@Test
func repairDetectorRequiresRepairWhenArtifactsAreMissing() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let detector = RepairDetector(
        installStateStore: InstallStateStore(paths: AppPaths(root: root))
    )

    #expect(try detector.requiresRepair())
}
