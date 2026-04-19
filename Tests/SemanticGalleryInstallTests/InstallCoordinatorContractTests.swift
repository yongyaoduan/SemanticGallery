import Foundation
import Testing
@testable import SemanticGalleryInstall
@testable import SemanticGalleryPersistence

private actor InstallProgressRecorder {
    private var items: [InstallProgress] = []

    func append(_ item: InstallProgress) {
        items.append(item)
    }

    func values() -> [InstallProgress] {
        items
    }
}

@Test
func installCoordinatorPrepareProducesOneTerminalSnapshotPerStepAndMarksInstallationComplete() async throws {
    /// Formal specification for callers:
    /// Precondition:
    ///   1. `paths.root` is writable.
    ///   2. `downloader` can provide every file required by `catalog`.
    /// Postcondition:
    ///   1. `snapshot.map(\\.step) = InstallStep.allCases`.
    ///   2. `snapshot.last?.overallProgress = 1`.
    ///   3. `InstallationVerifier(paths: paths, catalog: catalog).isInstallationComplete() = true`.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let catalog = ArtifactCatalog.legacyCompatible
    let coordinator = InstallCoordinator(
        paths: paths,
        downloader: ArtifactDownloader.stubbed,
        catalog: catalog
    )

    let snapshot = try await coordinator.prepare()

    #expect(snapshot.map(\.step) == InstallStep.allCases)
    #expect(snapshot.last?.overallProgress == 1.0)
    #expect(snapshot.allSatisfy { $0.recordedAt != nil })
    #expect(try InstallationVerifier(paths: paths, catalog: catalog).isInstallationComplete())
}

@Test
func installCoordinatorPrepareReturnsTheLatestStreamedProgressForEachStep() async throws {
    /// Formal specification for callers:
    /// Precondition:
    ///   1. `prepare(onProgress:)` is awaited to completion.
    /// Postcondition:
    ///   Let `stream` be the sequence delivered to `onProgress`.
    ///   For every `step ∈ InstallStep.allCases`, the returned snapshot contains exactly the last
    ///   element of `stream` whose `step` equals `step`.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let recorder = InstallProgressRecorder()
    let coordinator = InstallCoordinator(
        paths: AppPaths(root: root),
        downloader: ArtifactDownloader.stubbed
    )

    let snapshot = try await coordinator.prepare { item in
        await recorder.append(item)
    }
    let streamed = await recorder.values()
    let latestByStep = Dictionary(grouping: streamed, by: \.step).compactMapValues(\.last)
    let expected = InstallStep.allCases.compactMap { latestByStep[$0] }

    #expect(snapshot == expected)
}
