import Foundation
import Testing
@testable import SemanticGalleryInstall
@testable import SemanticGalleryPersistence

@Test
func incompleteInstallReturnsFalse() throws {
    /// Formal specification for callers:
    /// Pre: no required installation artifacts exist under `paths.runtimeArtifactsRoot`.
    /// Post after `isInstallationComplete()`:
    /// the result is `false`.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let store = InstallationVerifier(paths: paths)
    #expect(try store.isInstallationComplete() == false)
}

@Test
func installStateRequiresEveryExpectedArtifactFile() async throws {
    /// Formal specification for callers:
    /// Pre: every required artifact exists and validates.
    /// Post after deleting any required artifact file and calling `isInstallationComplete()`:
    /// the result is `false`.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()

    let store = InstallationVerifier(paths: paths)
    #expect(try store.isInstallationComplete())

    let missingFileURL = paths.artifactsRoot
        .appending(path: ArtifactCatalog.legacyCompatible.stage1Checkpoint.relativePath)
        .appending(path: "summary.json")
    try FileManager.default.removeItem(at: missingFileURL)

    #expect(try store.isInstallationComplete() == false)
}

@Test
func installStateRejectsCorruptArtifactPayloads() async throws {
    /// Formal specification for callers:
    /// Pre: every required artifact exists.
    /// Post after corrupting a validated JSON payload and calling `isInstallationComplete()`:
    /// the result is `false`.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()

    let store = InstallationVerifier(paths: paths)
    #expect(try store.isInstallationComplete())

    let corruptConfigURL = paths.artifactsRoot
        .appending(path: ArtifactCatalog.legacyCompatible.baseModel.relativePath)
        .appending(path: "config.json")
    try Data("not-json".utf8).write(to: corruptConfigURL)

    #expect(try store.isInstallationComplete() == false)
}

@Test
func bundledArtifactsCountAsCompleteWithoutInstallStateFile() async throws {
    /// Formal specification for callers:
    /// Pre: `paths.runtimeArtifactsRoot = paths.bundledArtifactsRoot`, and bundled artifacts validate.
    /// Post after `isInstallationComplete()`:
    /// the result is `true` even when `installStateURL` is absent.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let bundledRoot = root.appending(path: "BundledArtifacts")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let installCoordinator = InstallCoordinator(
        paths: AppPaths(root: root),
        downloader: ArtifactDownloader.stubbed
    )
    _ = try await installCoordinator.prepare()

    try FileManager.default.copyItem(
        at: AppPaths(root: root).artifactsRoot,
        to: bundledRoot
    )
    try FileManager.default.removeItem(at: AppPaths(root: root).installStateURL)

    let paths = AppPaths(
        supportRoot: root.appending(path: "Runtime"),
        cachesRoot: root.appending(path: "Caches"),
        logsRoot: root.appending(path: "Logs"),
        bundledArtifactsRoot: bundledRoot
    )
    let store = InstallationVerifier(paths: paths)

    #expect(try store.isInstallationComplete())
}
