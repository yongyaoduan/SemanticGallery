import Foundation
import Testing
@testable import SemanticGalleryPersistence
@testable import SemanticGallerySearch

@Test
func searchCoordinatorReturnsVisibleFolderItemsForEmptyQueryAndFiltersTextMatches() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)
    let firstAssetID = try database.upsertAsset(contentHash: "hash-a", fileSize: 128, pixelWidth: nil, pixelHeight: nil)
    let secondAssetID = try database.upsertAsset(contentHash: "hash-b", fileSize: 256, pixelWidth: nil, pixelHeight: nil)

    try database.upsertFileInstance(
        assetID: firstAssetID,
        folderID: folderID,
        absolutePath: "/tmp/library/Trips/one.jpg",
        relativePath: "Trips/one.jpg",
        mtimeNanoseconds: 42,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: secondAssetID,
        folderID: folderID,
        absolutePath: "/tmp/library/Portraits/two.jpg",
        relativePath: "Portraits/two.jpg",
        mtimeNanoseconds: 84,
        isPresent: true
    )

    let coordinator = SearchCoordinator(database: database)
    let allResults = try coordinator.search(folderAbsolutePath: "/tmp/library", query: "", limit: 25)
    let filteredResults = try coordinator.search(folderAbsolutePath: "/tmp/library", query: "trip", limit: 25)

    #expect(allResults.map(\.absolutePath) == [
        "/tmp/library/Portraits/two.jpg",
        "/tmp/library/Trips/one.jpg",
    ])
    #expect(filteredResults.map(\.absolutePath) == ["/tmp/library/Trips/one.jpg"])
}

@Test
@MainActor
func selectionStateSupportsToggleSelectAllAndClear() {
    let items = [
        SearchAssetRecord(assetID: 1, absolutePath: "/tmp/library/one.jpg", thumbnailPath: nil),
        SearchAssetRecord(assetID: 2, absolutePath: "/tmp/library/two.jpg", thumbnailPath: nil),
    ]
    let state = SelectionState()

    state.toggle(assetID: 1)
    #expect(state.selectedAssetIDs == [1])

    state.selectAll(items: items)
    #expect(state.selectedAssetIDs == [1, 2])

    state.clear()
    #expect(state.selectedAssetIDs.isEmpty)
}

@Test
func searchCoordinatorBuildsFolderScopedVectorViewAndFindsSimilarImages() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderA = try database.upsertFolder(absolutePath: "/tmp/folder-a", bookmarkData: nil, isActive: true)
    let folderB = try database.upsertFolder(absolutePath: "/tmp/folder-b", bookmarkData: nil, isActive: false)
    let firstAssetID = try database.upsertAsset(contentHash: "hash-a", fileSize: 128, pixelWidth: nil, pixelHeight: nil)
    let secondAssetID = try database.upsertAsset(contentHash: "hash-b", fileSize: 256, pixelWidth: nil, pixelHeight: nil)
    let thirdAssetID = try database.upsertAsset(contentHash: "hash-c", fileSize: 512, pixelWidth: nil, pixelHeight: nil)

    try database.upsertFileInstance(
        assetID: firstAssetID,
        folderID: folderA,
        absolutePath: "/tmp/folder-a/one.jpg",
        relativePath: "one.jpg",
        mtimeNanoseconds: 10,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: secondAssetID,
        folderID: folderA,
        absolutePath: "/tmp/folder-a/two.jpg",
        relativePath: "two.jpg",
        mtimeNanoseconds: 11,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: thirdAssetID,
        folderID: folderB,
        absolutePath: "/tmp/folder-b/other.jpg",
        relativePath: "other.jpg",
        mtimeNanoseconds: 12,
        isPresent: true
    )

    try database.upsertEmbedding(assetID: firstAssetID, encoderVersion: "stage1", vector: [1.0, 0.0])
    try database.upsertEmbedding(assetID: secondAssetID, encoderVersion: "stage1", vector: [0.8, 0.2])
    try database.upsertEmbedding(assetID: thirdAssetID, encoderVersion: "stage1", vector: [0.0, 1.0])

    let coordinator = SearchCoordinator(database: database, encoderVersion: "stage1")
    let activeView = try coordinator.activeSearchView(folderAbsolutePath: "/tmp/folder-a")
    let vectorMatches = coordinator.vectorSearch(activeView: activeView, queryVector: [1.0, 0.0], limit: 10)
    let similarMatches = try coordinator.searchSimilar(
        folderAbsolutePath: "/tmp/folder-a",
        recordID: activeView.items[0].id,
        limit: 10
    )

    #expect(activeView.items.map(\.absolutePath) == [
        "/tmp/folder-a/one.jpg",
        "/tmp/folder-a/two.jpg",
    ])
    #expect(vectorMatches.map(\.absolutePath) == [
        "/tmp/folder-a/one.jpg",
        "/tmp/folder-a/two.jpg",
    ])
    #expect(similarMatches.map(\.absolutePath) == ["/tmp/folder-a/two.jpg"])
}
