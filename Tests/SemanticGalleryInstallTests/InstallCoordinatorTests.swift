import Foundation
import Testing
@testable import SemanticGalleryInstall
@testable import SemanticGalleryPersistence

private actor InstallStepRecorder {
    private var items: [InstallProgress] = []

    func append(_ item: InstallProgress) {
        items.append(item)
    }

    func values() -> [InstallProgress] {
        items
    }
}

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
func installCoordinatorStreamsProgressRowsInOrder() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let coordinator = InstallCoordinator(
        paths: AppPaths(root: root),
        downloader: ArtifactDownloader.stubbed
    )

    let recorder = InstallStepRecorder()
    let steps = try await coordinator.prepare { item in
        await recorder.append(item)
    }.map(\.step)
    let streamedSteps = await recorder.values().map(\.step)
    let streamedUniqueSteps = streamedSteps.reduce(into: [InstallStep]()) { partialResult, step in
        if partialResult.last != step {
            partialResult.append(step)
        }
    }

    #expect(streamedUniqueSteps == steps)
}

@Test
func installCoordinatorReportsArtifactFileCountsAndTiming() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let coordinator = InstallCoordinator(
        paths: AppPaths(root: root),
        downloader: ArtifactDownloader.stubbed
    )

    let recorder = InstallStepRecorder()
    _ = try await coordinator.prepare { item in
        await recorder.append(item)
    }
    let streamedItems = await recorder.values()

    let baseModelItems = streamedItems.filter { $0.step == .downloadBaseModel }
    let stage1Items = streamedItems.filter { $0.step == .downloadStage1Checkpoint }
    let publicAnchorItems = streamedItems.filter { $0.step == .downloadPublicAnchor }

    #expect(baseModelItems.count == ArtifactCatalog.legacyCompatible.baseModel.requiredFiles.count + 1)
    #expect(stage1Items.count == ArtifactCatalog.legacyCompatible.stage1Checkpoint.requiredFiles.count + 1)
    #expect(publicAnchorItems.count == ArtifactCatalog.legacyCompatible.publicAnchor.requiredFiles.count + 1)
    #expect(baseModelItems.first?.message == "Preparing the shared config, preprocessor, and tokenizer files")
    #expect(baseModelItems.last?.message.contains("Downloaded 5 of 5 files") == true)
    #expect(stage1Items.last?.message.contains("Downloaded 2 of 2 files") == true)
    #expect(publicAnchorItems.last?.message.contains("Downloaded 2 of 2 files") == true)
    #expect(baseModelItems.last?.stepProgress == 1.0)
    #expect(publicAnchorItems.last?.overallProgress == 11.0 / 13.0)
}

@Test
func installationVerifierRejectsMissingArtifacts() throws {
    /// Formal specification for callers:
    /// Pre: no required artifact files exist under `paths.runtimeArtifactsRoot`.
    /// Post after `isInstallationComplete()`:
    /// the result is `false`.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let verifier = InstallationVerifier(paths: AppPaths(root: root))

    #expect(try verifier.isInstallationComplete() == false)
}

@Test
func installCoordinatorReplacesIncompleteArtifactsBeforePreparing() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let staleSummaryURL = paths.artifactsRoot
        .appending(path: ArtifactCatalog.legacyCompatible.stage1Checkpoint.relativePath)
        .appending(path: "summary.json")
    try FileManager.default.createDirectory(at: staleSummaryURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("stale".utf8).write(to: staleSummaryURL)

    let coordinator = InstallCoordinator(
        paths: paths,
        downloader: ArtifactDownloader.stubbed
    )

    _ = try await coordinator.prepare()

    let installationVerifier = InstallationVerifier(paths: paths)
    #expect(try installationVerifier.isInstallationComplete())

    let summaryData = try Data(contentsOf: staleSummaryURL)
    let summary = try #require(try JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
    #expect(summary["source"] as? String == "stub")
}
