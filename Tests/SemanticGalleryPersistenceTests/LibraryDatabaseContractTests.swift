import Foundation
import Testing
@testable import SemanticGalleryPersistence

@Suite("Library Database Contracts")
struct LibraryDatabaseContractTests {
    /// Formal specification for callers:
    /// Pre: folder `f` exists, and `summaryJSON = nil`.
    /// Post after `insertTrainingRun(..., summaryJSON: nil)` followed by
    /// `finishTrainingRun(..., summaryJSON: nil)`:
    /// the persisted run keeps `summaryJSON' = nil`.
    @Test
    func trainingRunsPreserveNilSummariesAcrossInsertAndFinish() throws {
        let database = try makeDatabase()
        let folderID = try database.upsertFolder(
            absolutePath: "/tmp/library",
            bookmarkData: nil,
            isActive: true
        )

        let runID = try database.insertTrainingRun(
            folderID: folderID,
            encoderVersion: "stage1",
            status: "running",
            summaryJSON: nil
        )
        try database.finishTrainingRun(
            id: runID,
            status: "completed",
            summaryJSON: nil
        )

        let runs = try database.trainingRuns(inFolderAbsolutePath: "/tmp/library")
        let run = try #require(runs.first)
        #expect(run.summaryJSON == nil)
    }

    /// Formal specification for callers:
    /// Pre: folder `f` contains file-instance paths `K ∪ M`, and the caller
    /// passes `keepingAbsolutePaths = K`.
    /// Post after `markMissingFileInstances(inFolderAbsolutePath: f, keepingAbsolutePaths: K)`:
    /// exactly the paths in `M` are removed, and each path in `K` remains queryable.
    @Test
    func markMissingFileInstancesRemovesOnlyPathsOutsideTheCallerProvidedKeepSet() throws {
        let database = try makeDatabase()
        let folderID = try database.upsertFolder(
            absolutePath: "/tmp/library",
            bookmarkData: nil,
            isActive: true
        )
        let sharedAssetID = try database.upsertAsset(
            contentHash: "hash-shared",
            fileSize: 128,
            pixelWidth: nil,
            pixelHeight: nil
        )
        let uniqueAssetID = try database.upsertAsset(
            contentHash: "hash-unique",
            fileSize: 256,
            pixelWidth: nil,
            pixelHeight: nil
        )

        try database.upsertFileInstance(
            assetID: sharedAssetID,
            folderID: folderID,
            absolutePath: "/tmp/library/keep.jpg",
            relativePath: "keep.jpg",
            mtimeNanoseconds: 1,
            isPresent: true
        )
        try database.upsertFileInstance(
            assetID: sharedAssetID,
            folderID: folderID,
            absolutePath: "/tmp/library/remove-shared.jpg",
            relativePath: "remove-shared.jpg",
            mtimeNanoseconds: 2,
            isPresent: true
        )
        try database.upsertFileInstance(
            assetID: uniqueAssetID,
            folderID: folderID,
            absolutePath: "/tmp/library/remove-unique.jpg",
            relativePath: "remove-unique.jpg",
            mtimeNanoseconds: 3,
            isPresent: true
        )
        try database.upsertEmbedding(
            assetID: sharedAssetID,
            encoderVersion: "stage1",
            vector: [1.0, 0.0]
        )

        try database.markMissingFileInstances(
            inFolderAbsolutePath: "/tmp/library",
            keepingAbsolutePaths: ["/tmp/library/keep.jpg"]
        )

        let visible = try database.visibleFileInstances(
            inFolderAbsolutePath: "/tmp/library",
            limit: 10
        )
        let embedded = try database.embeddedFiles(
            inFolderAbsolutePath: "/tmp/library",
            encoderVersion: "stage1"
        )

        #expect(visible.map(\.absolutePath) == ["/tmp/library/keep.jpg"])
        #expect(embedded.map(\.absolutePath) == ["/tmp/library/keep.jpg"])
    }

    /// Formal specification for callers:
    /// Pre: assets may appear in multiple folders, and none has an embedding for
    /// encoder `e`.
    /// Post after `missingEmbeddingWork(encoderVersion: e)`:
    /// the result contains exactly one representative record per asset.
    @Test
    func missingEmbeddingWorkAcrossAllFoldersReturnsOneRepresentativePerAsset() throws {
        let database = try makeDatabase()
        let folderA = try database.upsertFolder(
            absolutePath: "/tmp/library-a",
            bookmarkData: nil,
            isActive: true
        )
        let folderB = try database.upsertFolder(
            absolutePath: "/tmp/library-b",
            bookmarkData: nil,
            isActive: false
        )
        let sharedAssetID = try database.upsertAsset(
            contentHash: "hash-shared",
            fileSize: 128,
            pixelWidth: nil,
            pixelHeight: nil
        )
        let otherAssetID = try database.upsertAsset(
            contentHash: "hash-other",
            fileSize: 256,
            pixelWidth: nil,
            pixelHeight: nil
        )

        try database.upsertFileInstance(
            assetID: sharedAssetID,
            folderID: folderA,
            absolutePath: "/tmp/library-a/shared-a.jpg",
            relativePath: "shared-a.jpg",
            mtimeNanoseconds: 1,
            isPresent: true
        )
        try database.upsertFileInstance(
            assetID: sharedAssetID,
            folderID: folderB,
            absolutePath: "/tmp/library-b/shared-b.jpg",
            relativePath: "shared-b.jpg",
            mtimeNanoseconds: 2,
            isPresent: true
        )
        try database.upsertFileInstance(
            assetID: otherAssetID,
            folderID: folderB,
            absolutePath: "/tmp/library-b/other.jpg",
            relativePath: "other.jpg",
            mtimeNanoseconds: 3,
            isPresent: true
        )

        let work = try database.missingEmbeddingWork(encoderVersion: "stage1")

        #expect(work.count == 2)
        #expect(work.map(\.assetID).sorted() == [otherAssetID, sharedAssetID].sorted())
    }

    /// Formal specification for callers:
    /// Pre: two `upsertAsset` calls use the same `contentHash = h`.
    /// Post: both calls return the same persistent asset identifier.
    @Test
    func upsertAssetUsesTheContentHashAsTheStableAssetIdentity() throws {
        let database = try makeDatabase()

        let firstID = try database.upsertAsset(
            contentHash: "hash-shared",
            fileSize: 128,
            pixelWidth: 100,
            pixelHeight: 200
        )
        let secondID = try database.upsertAsset(
            contentHash: "hash-shared",
            fileSize: 512,
            pixelWidth: 400,
            pixelHeight: 800
        )

        #expect(firstID == secondID)
    }

    /// Formal specification for callers:
    /// Pre: asset `a` appears as file instances `p` and `q` in different folders,
    /// and `a` already has an embedding for encoder `e`.
    /// Post after `removeFileInstances([p])`:
    /// `q` remains visible in its folder, and `a` does not reappear in `missingEmbeddingWork(e)`.
    @Test
    func removingOneFileInstancePreservesSharedEmbeddingsForOtherFolders() throws {
        let database = try makeDatabase()
        let folderA = try database.upsertFolder(
            absolutePath: "/tmp/library-a",
            bookmarkData: nil,
            isActive: true
        )
        let folderB = try database.upsertFolder(
            absolutePath: "/tmp/library-b",
            bookmarkData: nil,
            isActive: false
        )
        let assetID = try database.upsertAsset(
            contentHash: "hash-shared",
            fileSize: 128,
            pixelWidth: nil,
            pixelHeight: nil
        )

        try database.upsertFileInstance(
            assetID: assetID,
            folderID: folderA,
            absolutePath: "/tmp/library-a/shared.jpg",
            relativePath: "shared.jpg",
            mtimeNanoseconds: 1,
            isPresent: true
        )
        try database.upsertFileInstance(
            assetID: assetID,
            folderID: folderB,
            absolutePath: "/tmp/library-b/shared.jpg",
            relativePath: "shared.jpg",
            mtimeNanoseconds: 2,
            isPresent: true
        )
        try database.upsertEmbedding(
            assetID: assetID,
            encoderVersion: "stage1",
            vector: [1.0, 0.0]
        )

        try database.removeFileInstances(absolutePaths: ["/tmp/library-a/shared.jpg"])

        let remaining = try database.visibleFileInstances(
            inFolderAbsolutePath: "/tmp/library-b",
            limit: 10
        )
        let missing = try database.missingEmbeddingWork(encoderVersion: "stage1")

        #expect(remaining.map(\.absolutePath) == ["/tmp/library-b/shared.jpg"])
        #expect(missing.isEmpty)
    }

    private func makeDatabase() throws -> LibraryDatabase {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
        try database.migrate()
        return database
    }
}
