import Foundation
import SQLite3
import Testing
@testable import SemanticGalleryPersistence

@Test
func libraryDatabaseCreatesPhaseBSchema() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()

    let tableNames = try Set(database.tableNames())
    #expect(tableNames.isSuperset(of: [
        "folders",
        "assets",
        "file_instances",
        "embeddings",
        "thumbnails",
        "training_runs",
        "install_state",
        "asset_terms",
    ]))
}

@Test
func libraryDatabaseEnablesConcurrentReadWriteSettings() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let databaseURL = root.appending(path: "library.sqlite")
    let database = try LibraryDatabase.open(at: databaseURL)
    try database.migrate()

    #expect(try pragmaValue(named: "journal_mode", in: databaseURL)?.lowercased() == "wal")
    _ = database
}

@Test
func libraryDatabaseWaitsForShortLivedWriteLocks() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let databaseURL = root.appending(path: "library.sqlite")
    let database = try LibraryDatabase.open(at: databaseURL)
    try database.migrate()

    var lockHandle: OpaquePointer?
    guard sqlite3_open(databaseURL.path(percentEncoded: false), &lockHandle) == SQLITE_OK, let lockHandle else {
        sqlite3_close(lockHandle)
        throw NSError(
            domain: "LibraryDatabaseTests",
            code: 3,
            userInfo: [NSLocalizedDescriptionKey: "Unable to open the lock-holding database connection."]
        )
    }
    defer { sqlite3_close(lockHandle) }

    guard sqlite3_exec(lockHandle, "BEGIN EXCLUSIVE TRANSACTION;", nil, nil, nil) == SQLITE_OK else {
        throw NSError(
            domain: "LibraryDatabaseTests",
            code: 4,
            userInfo: [NSLocalizedDescriptionKey: "Unable to begin the lock-holding transaction."]
        )
    }

    let releaseSemaphore = DispatchSemaphore(value: 0)
    let lockHandleValue = UInt(bitPattern: lockHandle)
    DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(300)) {
        if let commitHandle = OpaquePointer(bitPattern: lockHandleValue) {
            sqlite3_exec(commitHandle, "COMMIT;", nil, nil, nil)
        }
        releaseSemaphore.signal()
    }

    let start = Date()
    _ = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)
    let elapsed = Date().timeIntervalSince(start)
    releaseSemaphore.wait()
    let busyTimeoutMilliseconds = Double(try pragmaValue(named: "busy_timeout", in: databaseURL) ?? "0") ?? 0
    let deadline = (busyTimeoutMilliseconds / 1000.0) + 2.0

    #expect(elapsed >= 0.25)
    #expect(elapsed < deadline)
}

@Test
func libraryDatabaseStoresFileInstancesForSelectedFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: Data("bookmark".utf8), isActive: true)
    let assetID = try database.upsertAsset(contentHash: "hash-a", fileSize: 128, pixelWidth: 640, pixelHeight: 480)

    try database.upsertFileInstance(
        assetID: assetID,
        folderID: folderID,
        absolutePath: "/tmp/library/one.jpg",
        relativePath: "one.jpg",
        mtimeNanoseconds: 42,
        isPresent: true
    )

    let rows = try database.fileInstances(inFolderAbsolutePath: "/tmp/library")
    #expect(rows.count == 1)
    #expect(rows.first?.absolutePath == "/tmp/library/one.jpg")
    #expect(rows.first?.relativePath == "one.jpg")
}

@Test
func libraryDatabaseSearchesRelativePathsWithinActiveFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)
    let firstAssetID = try database.upsertAsset(contentHash: "hash-a", fileSize: 128, pixelWidth: 640, pixelHeight: 480)
    let secondAssetID = try database.upsertAsset(contentHash: "hash-b", fileSize: 256, pixelWidth: 640, pixelHeight: 480)

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

    let matches = try database.searchFileInstances(inFolderAbsolutePath: "/tmp/library", query: "trip", limit: 10)
    #expect(matches.map(\.relativePath) == ["Trips/one.jpg"])
}

@Test
func libraryDatabaseFallsBackToSubstringSearchForFilenameTokens() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)
    let firstAssetID = try database.upsertAsset(contentHash: "hash-c", fileSize: 128, pixelWidth: nil, pixelHeight: nil)
    let secondAssetID = try database.upsertAsset(contentHash: "hash-d", fileSize: 256, pixelWidth: nil, pixelHeight: nil)

    try database.upsertFileInstance(
        assetID: firstAssetID,
        folderID: folderID,
        absolutePath: "/tmp/library/sample-1.jpg",
        relativePath: "sample-1.jpg",
        mtimeNanoseconds: 10,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: secondAssetID,
        folderID: folderID,
        absolutePath: "/tmp/library/sample-2.jpg",
        relativePath: "sample-2.jpg",
        mtimeNanoseconds: 11,
        isPresent: true
    )

    let matches = try database.searchFileInstances(inFolderAbsolutePath: "/tmp/library", query: "sample-1", limit: 10)
    #expect(matches.map(\.relativePath) == ["sample-1.jpg"])
}

@Test
func libraryDatabaseReturnsMissingEmbeddingWorkOncePerAssetWithinFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)
    let assetID = try database.upsertAsset(contentHash: "hash-a", fileSize: 128, pixelWidth: 640, pixelHeight: 480)

    try database.upsertFileInstance(
        assetID: assetID,
        folderID: folderID,
        absolutePath: "/tmp/library/A/one.jpg",
        relativePath: "A/one.jpg",
        mtimeNanoseconds: 42,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: assetID,
        folderID: folderID,
        absolutePath: "/tmp/library/B/two.jpg",
        relativePath: "B/two.jpg",
        mtimeNanoseconds: 84,
        isPresent: true
    )

    let work = try database.missingEmbeddingWork(
        inFolderAbsolutePath: "/tmp/library",
        encoderVersion: "stage1"
    )

    #expect(work.count == 1)
    #expect(work.first?.assetID == assetID)
    #expect(work.first?.representativeAbsolutePath == "/tmp/library/A/one.jpg")
}

@Test
func libraryDatabaseCountsVisibleEmbeddedFilesForSelectedFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderA = try database.upsertFolder(absolutePath: "/tmp/library-a", bookmarkData: nil, isActive: true)
    let folderB = try database.upsertFolder(absolutePath: "/tmp/library-b", bookmarkData: nil, isActive: false)
    let sharedAssetID = try database.upsertAsset(contentHash: "hash-shared", fileSize: 128, pixelWidth: nil, pixelHeight: nil)
    let pendingAssetID = try database.upsertAsset(contentHash: "hash-pending", fileSize: 256, pixelWidth: nil, pixelHeight: nil)
    let otherFolderAssetID = try database.upsertAsset(contentHash: "hash-other", fileSize: 512, pixelWidth: nil, pixelHeight: nil)

    try database.upsertFileInstance(
        assetID: sharedAssetID,
        folderID: folderA,
        absolutePath: "/tmp/library-a/A/one.jpg",
        relativePath: "A/one.jpg",
        mtimeNanoseconds: 10,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: sharedAssetID,
        folderID: folderA,
        absolutePath: "/tmp/library-a/B/two.jpg",
        relativePath: "B/two.jpg",
        mtimeNanoseconds: 11,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: pendingAssetID,
        folderID: folderA,
        absolutePath: "/tmp/library-a/three.jpg",
        relativePath: "three.jpg",
        mtimeNanoseconds: 12,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: otherFolderAssetID,
        folderID: folderB,
        absolutePath: "/tmp/library-b/other.jpg",
        relativePath: "other.jpg",
        mtimeNanoseconds: 13,
        isPresent: true
    )

    try database.upsertEmbedding(assetID: sharedAssetID, encoderVersion: "stage1", vector: [1.0, 0.0])
    try database.upsertEmbedding(assetID: otherFolderAssetID, encoderVersion: "stage1", vector: [0.0, 1.0])

    let embeddedCount = try database.embeddedFileCount(
        inFolderAbsolutePath: "/tmp/library-a",
        encoderVersion: "stage1"
    )

    #expect(embeddedCount == 2)
}

@Test
func libraryDatabaseRemovesDeletedFileInstancesAndOrphanEmbeddings() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)
    let firstAssetID = try database.upsertAsset(contentHash: "hash-e", fileSize: 128, pixelWidth: nil, pixelHeight: nil)
    let secondAssetID = try database.upsertAsset(contentHash: "hash-f", fileSize: 256, pixelWidth: nil, pixelHeight: nil)

    try database.upsertFileInstance(
        assetID: firstAssetID,
        folderID: folderID,
        absolutePath: "/tmp/library/one.jpg",
        relativePath: "one.jpg",
        mtimeNanoseconds: 1,
        isPresent: true
    )
    try database.upsertFileInstance(
        assetID: secondAssetID,
        folderID: folderID,
        absolutePath: "/tmp/library/two.jpg",
        relativePath: "two.jpg",
        mtimeNanoseconds: 2,
        isPresent: true
    )
    try database.upsertEmbedding(assetID: firstAssetID, encoderVersion: "stage1", vector: [1.0, 0.0])
    try database.upsertEmbedding(assetID: secondAssetID, encoderVersion: "stage1", vector: [0.0, 1.0])

    try database.removeFileInstances(absolutePaths: ["/tmp/library/one.jpg"])

    let visibleRows = try database.visibleFileInstances(inFolderAbsolutePath: "/tmp/library", limit: 10)
    let allRows = try database.fileInstances(inFolderAbsolutePath: "/tmp/library")
    let deletedEmbedding = try database.embeddingVector(assetID: firstAssetID, encoderVersion: "stage1")
    let remainingEmbedding = try database.embeddingVector(assetID: secondAssetID, encoderVersion: "stage1")

    #expect(visibleRows.map(\.relativePath) == ["two.jpg"])
    #expect(allRows.map(\.relativePath) == ["two.jpg"])
    #expect(deletedEmbedding == nil)
    #expect(remainingEmbedding != nil)
}

@Test
func libraryDatabaseStoresTrainingRunsPerFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let database = try LibraryDatabase.open(at: root.appending(path: "library.sqlite"))
    try database.migrate()
    let folderID = try database.upsertFolder(absolutePath: "/tmp/library", bookmarkData: nil, isActive: true)

    let runID = try database.insertTrainingRun(
        folderID: folderID,
        encoderVersion: "stage2-library-abcdef",
        status: "running",
        summaryJSON: #"{"phase":"prepare"}"#
    )

    try database.finishTrainingRun(
        id: runID,
        status: "completed",
        summaryJSON: #"{"phase":"reindex","encoder_version":"stage2-library-abcdef"}"#
    )

    let runs = try database.trainingRuns(inFolderAbsolutePath: "/tmp/library")
    #expect(runs.count == 1)
    #expect(runs.first?.encoderVersion == "stage2-library-abcdef")
    #expect(runs.first?.status == "completed")
    #expect(runs.first?.summaryJSON?.contains("reindex") == true)
}

private func pragmaValue(named name: String, in databaseURL: URL) throws -> String? {
    var handle: OpaquePointer?
    guard sqlite3_open(databaseURL.path(percentEncoded: false), &handle) == SQLITE_OK, let handle else {
        sqlite3_close(handle)
        throw NSError(
            domain: "LibraryDatabaseTests",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Unable to open the library database."]
        )
    }
    defer { sqlite3_close(handle) }

    var statement: OpaquePointer?
    let sql = "PRAGMA \(name);"
    guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
        sqlite3_finalize(statement)
        throw NSError(
            domain: "LibraryDatabaseTests",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Unable to prepare the database pragma query."]
        )
    }
    defer { sqlite3_finalize(statement) }

    guard sqlite3_step(statement) == SQLITE_ROW else {
        return nil
    }

    if let textPointer = sqlite3_column_text(statement, 0) {
        return String(cString: textPointer)
    }

    return String(sqlite3_column_int(statement, 0))
}
