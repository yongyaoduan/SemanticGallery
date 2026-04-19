import Foundation
import Testing
@testable import SemanticGalleryIndex
@testable import SemanticGalleryPersistence

@Suite("Folder Indexer Contracts")
struct FolderIndexerContractTests {
    /// Formal specification for callers:
    /// Pre: folder `f` contains two supported image files `p` and `q` with identical bytes.
    /// Post after `rebuildIndex(for: f)`:
    /// `summary.fileCount = 2`, `summary.uniqueAssetCount = 1`, and both caller-visible paths remain queryable.
    @Test
    func rebuildIndexDeduplicatesEqualContentWithoutDroppingAnyVisiblePath() async throws {
        let runtimeRoot = try makeRuntimeRoot()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        let gallery = runtimeRoot.appending(path: "Gallery")
        try FileManager.default.createDirectory(at: gallery, withIntermediateDirectories: true)
        let sharedBytes = Data("shared-image".utf8)
        try sharedBytes.write(to: gallery.appending(path: "first.jpg"))
        try sharedBytes.write(to: gallery.appending(path: "second.jpg"))

        let database = try makeDatabase(in: runtimeRoot)
        let indexer = FolderIndexer(database: database)

        let summary = try await indexer.rebuildIndex(for: gallery)
        let rows = try database.fileInstances(inFolderAbsolutePath: gallery.path(percentEncoded: false))

        #expect(summary.fileCount == 2)
        #expect(summary.uniqueAssetCount == 1)
        #expect(rows.map(\.relativePath) == ["first.jpg", "second.jpg"])
        #expect(Set(rows.map(\.assetID)).count == 1)
    }

    /// Formal specification for callers:
    /// Pre: folder `f` contains a supported image file at nested relative path `r`.
    /// Post after `rebuildIndex(for: f)`:
    /// the persisted file instance remains queryable with `relativePath = r`.
    @Test
    func rebuildIndexPreservesNestedCallerVisibleRelativePaths() async throws {
        let runtimeRoot = try makeRuntimeRoot()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        let gallery = runtimeRoot.appending(path: "Gallery")
        let nestedFolder = gallery.appending(path: "Trips/Beach")
        try FileManager.default.createDirectory(at: nestedFolder, withIntermediateDirectories: true)
        let photo = nestedFolder.appending(path: "photo.png")
        try Data("nested-image".utf8).write(to: photo)

        let database = try makeDatabase(in: runtimeRoot)
        let indexer = FolderIndexer(database: database)

        _ = try await indexer.rebuildIndex(for: gallery)
        let rows = try database.fileInstances(inFolderAbsolutePath: gallery.path(percentEncoded: false))

        #expect(rows.map(\.relativePath) == ["Trips/Beach/photo.png"])
        #expect(
            rows.map { URL(filePath: $0.absolutePath).resolvingSymlinksInPath().path(percentEncoded: false) } ==
            [photo.resolvingSymlinksInPath().path(percentEncoded: false)]
        )
    }

    private func makeRuntimeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeDatabase(in runtimeRoot: URL) throws -> LibraryDatabase {
        let database = try LibraryDatabase.open(at: runtimeRoot.appending(path: "library.sqlite"))
        try database.migrate()
        return database
    }
}
