import Foundation
import Testing
@testable import SemanticGalleryIndex
@testable import SemanticGalleryPersistence

@Test
func supportedImagePathMatchesLegacySuffixRules() {
    #expect(SupportedImagePath(url: URL(filePath: "/tmp/photo.JPG"))?.url.path == "/tmp/photo.JPG")
    #expect(SupportedImagePath(url: URL(filePath: "/tmp/note.txt")) == nil)
}

@Test
func folderIndexerStoresOnlySupportedVisibleImages() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let gallery = root.appending(path: "Gallery")
    try FileManager.default.createDirectory(at: gallery, withIntermediateDirectories: true)
    try Data("visible".utf8).write(to: gallery.appending(path: "visible.jpg"))
    try Data("note".utf8).write(to: gallery.appending(path: "notes.txt"))
    try FileManager.default.createDirectory(at: gallery.appending(path: ".private"), withIntermediateDirectories: true)
    try Data("hidden".utf8).write(to: gallery.appending(path: ".private/hidden.png"))

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()

    let indexer = FolderIndexer(database: database)
    try await indexer.rebuildIndex(for: gallery)

    let rows = try database.fileInstances(inFolderAbsolutePath: gallery.path(percentEncoded: false))
    #expect(rows.count == 1)
    #expect(rows.first?.relativePath == "visible.jpg")
}

@Test
func folderIndexerRemovesMissingFilesFromTheIndexOnRebuild() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let gallery = root.appending(path: "Gallery")
    try FileManager.default.createDirectory(at: gallery, withIntermediateDirectories: true)

    let first = gallery.appending(path: "cover.jpg")
    let second = gallery.appending(path: "inside.png")
    try Data("cover".utf8).write(to: first)
    try Data("inside".utf8).write(to: second)

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()

    let indexer = FolderIndexer(database: database)
    try await indexer.rebuildIndex(for: gallery)

    try FileManager.default.removeItem(at: second)
    try await indexer.rebuildIndex(for: gallery)

    let visible = try database.visibleFileInstances(inFolderAbsolutePath: gallery.path(percentEncoded: false), limit: 10)
    let allRows = try database.fileInstances(inFolderAbsolutePath: gallery.path(percentEncoded: false))

    #expect(visible.map(\.relativePath) == ["cover.jpg"])
    #expect(allRows.map(\.relativePath) == ["cover.jpg"])
}
