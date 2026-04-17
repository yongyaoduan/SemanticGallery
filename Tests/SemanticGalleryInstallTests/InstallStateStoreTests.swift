import Foundation
import Testing
@testable import SemanticGalleryInstall
@testable import SemanticGalleryPersistence

@Test
func incompleteInstallReturnsFalse() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let store = InstallStateStore(paths: paths)
    #expect(try store.isInstallationComplete() == false)
}

@Test
func installStateRequiresEveryExpectedArtifactFile() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()

    let store = InstallStateStore(paths: paths)
    #expect(try store.isInstallationComplete())

    let missingFileURL = paths.artifactsRoot
        .appending(path: ArtifactCatalog.legacyCompatible.stage1Checkpoint.relativePath)
        .appending(path: "summary.json")
    try FileManager.default.removeItem(at: missingFileURL)

    #expect(try store.isInstallationComplete() == false)
}

@Test
func installStateRejectsCorruptArtifactPayloads() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()

    let store = InstallStateStore(paths: paths)
    #expect(try store.isInstallationComplete())

    let corruptConfigURL = paths.artifactsRoot
        .appending(path: ArtifactCatalog.legacyCompatible.baseModel.relativePath)
        .appending(path: "config.json")
    try Data("not-json".utf8).write(to: corruptConfigURL)

    #expect(try store.isInstallationComplete() == false)
}

@Test
func bundledArtifactsCountAsCompleteWithoutInstallStateFile() async throws {
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
    let store = InstallStateStore(paths: paths)

    #expect(try store.isInstallationComplete())
}
