import Foundation
import Testing
@testable import SemanticGalleryPersistence
@testable import SemanticGallerySettings

@Test
func uninstallRemovesAppArtifactsButKeepsUserFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let userFolder = root.appendingPathComponent("UserAlbum")
    try FileManager.default.createDirectory(at: userFolder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let paths = AppPaths(root: root)
    try FileManager.default.createDirectory(at: paths.supportRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: paths.cachesRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: paths.logsRoot, withIntermediateDirectories: true)

    let coordinator = UninstallCoordinator()
    try coordinator.removeArtifacts(paths: paths, selectedFolder: userFolder)

    #expect(FileManager.default.fileExists(atPath: userFolder.path))
    #expect(FileManager.default.fileExists(atPath: paths.supportRoot.path) == false)
    #expect(FileManager.default.fileExists(atPath: paths.cachesRoot.path) == false)
    #expect(FileManager.default.fileExists(atPath: paths.logsRoot.path) == false)
}
