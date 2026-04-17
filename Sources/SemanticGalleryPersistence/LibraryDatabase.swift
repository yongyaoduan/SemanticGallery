import Foundation
import SQLite3

public struct FileInstanceRecord: Sendable, Equatable {
    public let fileInstanceID: Int64
    public let assetID: Int64
    public let absolutePath: String
    public let relativePath: String
    public let isPresent: Bool

    init(
        fileInstanceID: Int64,
        assetID: Int64,
        absolutePath: String,
        relativePath: String,
        isPresent: Bool
    ) {
        self.fileInstanceID = fileInstanceID
        self.assetID = assetID
        self.absolutePath = absolutePath
        self.relativePath = relativePath
        self.isPresent = isPresent
    }
}

public struct EmbeddingWorkRecord: Sendable, Equatable {
    public let assetID: Int64
    public let representativeAbsolutePath: String
    public let representativeRelativePath: String

    public init(assetID: Int64, representativeAbsolutePath: String, representativeRelativePath: String) {
        self.assetID = assetID
        self.representativeAbsolutePath = representativeAbsolutePath
        self.representativeRelativePath = representativeRelativePath
    }
}

public struct EmbeddedFileRecord: Sendable, Equatable {
    public let fileInstanceID: Int64
    public let assetID: Int64
    public let absolutePath: String
    public let relativePath: String
    public let vector: [Double]

    public init(
        fileInstanceID: Int64,
        assetID: Int64,
        absolutePath: String,
        relativePath: String,
        vector: [Double]
    ) {
        self.fileInstanceID = fileInstanceID
        self.assetID = assetID
        self.absolutePath = absolutePath
        self.relativePath = relativePath
        self.vector = vector
    }
}

public struct TrainingRunRecord: Sendable, Equatable {
    public let id: Int64
    public let folderID: Int64
    public let encoderVersion: String
    public let status: String
    public let startedAt: String?
    public let finishedAt: String?
    public let summaryJSON: String?

    init(
        id: Int64,
        folderID: Int64,
        encoderVersion: String,
        status: String,
        startedAt: String?,
        finishedAt: String?,
        summaryJSON: String?
    ) {
        self.id = id
        self.folderID = folderID
        self.encoderVersion = encoderVersion
        self.status = status
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.summaryJSON = summaryJSON
    }
}

public final class LibraryDatabase: @unchecked Sendable {
    private let handle: OpaquePointer

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        sqlite3_close(handle)
    }

    public static func open(at url: URL) throws -> LibraryDatabase {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var handle: OpaquePointer?
        if sqlite3_open_v2(
            url.path(percentEncoded: false),
            &handle,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            nil
        ) != SQLITE_OK {
            let message = handle.flatMap { sqlite3_errmsg($0).map { String(cString: $0) } } ?? "Unable to open database."
            if let handle {
                sqlite3_close(handle)
            }
            throw DatabaseError(message)
        }

        guard let handle else {
            throw DatabaseError("Unable to open database.")
        }

        do {
            try configureConnection(handle)
        } catch {
            sqlite3_close(handle)
            throw error
        }

        return LibraryDatabase(handle: handle)
    }

    public func migrate() throws {
        try executeScript(
            """
            CREATE TABLE IF NOT EXISTS folders(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              absolute_path TEXT NOT NULL UNIQUE,
              bookmark_data BLOB,
              is_active INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS assets(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              content_hash TEXT NOT NULL UNIQUE,
              file_size INTEGER NOT NULL,
              pixel_width INTEGER,
              pixel_height INTEGER,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS file_instances(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              asset_id INTEGER NOT NULL,
              folder_id INTEGER NOT NULL,
              absolute_path TEXT NOT NULL UNIQUE,
              relative_path TEXT NOT NULL,
              mtime_ns INTEGER NOT NULL,
              is_present INTEGER NOT NULL DEFAULT 1,
              deleted_at TEXT,
              FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
              FOREIGN KEY(folder_id) REFERENCES folders(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embeddings(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              asset_id INTEGER NOT NULL,
              encoder_version TEXT NOT NULL,
              vector_blob BLOB NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE,
              UNIQUE(asset_id, encoder_version)
            );

            CREATE TABLE IF NOT EXISTS thumbnails(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              asset_id INTEGER NOT NULL,
              cache_key TEXT NOT NULL UNIQUE,
              disk_path TEXT NOT NULL,
              pixel_width INTEGER NOT NULL,
              pixel_height INTEGER NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS training_runs(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              folder_id INTEGER NOT NULL,
              encoder_version TEXT NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              summary_json TEXT,
              FOREIGN KEY(folder_id) REFERENCES folders(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS install_state(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              app_version TEXT,
              schema_version TEXT,
              model_version TEXT,
              anchor_version TEXT,
              verified_at TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS asset_terms USING fts5(
              filename,
              relative_path,
              tokenize='unicode61'
            );
            """
        )
    }

    public func tableNames() throws -> [String] {
        try queryStrings(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'virtual table')
            ORDER BY name
            """
        )
    }

    public func upsertFolder(
        absolutePath: String,
        bookmarkData: Data?,
        isActive: Bool
    ) throws -> Int64 {
        try execute(
            """
            INSERT INTO folders (absolute_path, bookmark_data, is_active, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(absolute_path) DO UPDATE SET
              bookmark_data = excluded.bookmark_data,
              is_active = excluded.is_active,
              updated_at = CURRENT_TIMESTAMP
            """,
            bindings: [
                .text(absolutePath),
                .blob(bookmarkData),
                .integer(isActive ? 1 : 0),
            ]
        )

        let row = try queryRow(
            """
            SELECT id
            FROM folders
            WHERE absolute_path = ?
            """,
            bindings: [.text(absolutePath)]
        )
        defer { sqlite3_finalize(row) }
        return try requireInteger(row, at: 0)
    }

    public func containsFolder(absolutePath: String) throws -> Bool {
        let statement = try prepare(
            """
            SELECT 1
            FROM folders
            WHERE absolute_path = ?
            LIMIT 1
            """
        )
        defer { sqlite3_finalize(statement) }

        try bind(statement: statement, bindings: [.text(absolutePath)])
        return sqlite3_step(statement) == SQLITE_ROW
    }

    public func upsertAsset(
        contentHash: String,
        fileSize: Int64,
        pixelWidth: Int?,
        pixelHeight: Int?
    ) throws -> Int64 {
        try execute(
            """
            INSERT INTO assets (content_hash, file_size, pixel_width, pixel_height)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash) DO UPDATE SET
              file_size = excluded.file_size,
              pixel_width = excluded.pixel_width,
              pixel_height = excluded.pixel_height
            """,
            bindings: [
                .text(contentHash),
                .integer(fileSize),
                .integer(pixelWidth.map(Int64.init)),
                .integer(pixelHeight.map(Int64.init)),
            ]
        )

        let row = try queryRow(
            """
            SELECT id
            FROM assets
            WHERE content_hash = ?
            """,
            bindings: [.text(contentHash)]
        )
        defer { sqlite3_finalize(row) }
        return try requireInteger(row, at: 0)
    }

    public func upsertFileInstance(
        assetID: Int64,
        folderID: Int64,
        absolutePath: String,
        relativePath: String,
        mtimeNanoseconds: Int64,
        isPresent: Bool
    ) throws {
        try execute(
            """
            INSERT INTO file_instances (
              asset_id,
              folder_id,
              absolute_path,
              relative_path,
              mtime_ns,
              is_present,
              deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(absolute_path) DO UPDATE SET
              asset_id = excluded.asset_id,
              folder_id = excluded.folder_id,
              relative_path = excluded.relative_path,
              mtime_ns = excluded.mtime_ns,
              is_present = excluded.is_present,
              deleted_at = CASE excluded.is_present WHEN 1 THEN NULL ELSE CURRENT_TIMESTAMP END
            """,
            bindings: [
                .integer(assetID),
                .integer(folderID),
                .text(absolutePath),
                .text(relativePath),
                .integer(mtimeNanoseconds),
                .integer(isPresent ? 1 : 0),
            ]
        )

        try execute(
            """
            DELETE FROM asset_terms
            WHERE rowid IN (
              SELECT id
              FROM file_instances
              WHERE absolute_path = ?
            )
            """,
            bindings: [.text(absolutePath)]
        )

        try execute(
            """
            INSERT INTO asset_terms(rowid, filename, relative_path)
            SELECT id, ?, ?
            FROM file_instances
            WHERE absolute_path = ?
            """,
            bindings: [
                .text(URL(filePath: absolutePath).lastPathComponent),
                .text(relativePath),
                .text(absolutePath),
            ]
        )
    }

    public func fileInstances(inFolderAbsolutePath absolutePath: String) throws -> [FileInstanceRecord] {
        let statement = try prepare(
            """
            SELECT file_instances.id, file_instances.asset_id, file_instances.absolute_path, file_instances.relative_path, file_instances.is_present
            FROM file_instances
            JOIN folders ON folders.id = file_instances.folder_id
            WHERE folders.absolute_path = ?
            ORDER BY file_instances.absolute_path
            """
        )
        defer { sqlite3_finalize(statement) }

        try bind(statement: statement, bindings: [.text(absolutePath)])
        var records: [FileInstanceRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            records.append(
                FileInstanceRecord(
                    fileInstanceID: sqlite3_column_int64(statement, 0),
                    assetID: sqlite3_column_int64(statement, 1),
                    absolutePath: requireText(statement, at: 2),
                    relativePath: requireText(statement, at: 3),
                    isPresent: sqlite3_column_int64(statement, 4) != 0
                )
            )
        }
        return records
    }

    public func visibleFileInstances(
        inFolderAbsolutePath absolutePath: String,
        limit: Int
    ) throws -> [FileInstanceRecord] {
        let statement = try prepare(
            """
            SELECT file_instances.id, file_instances.asset_id, file_instances.absolute_path, file_instances.relative_path, file_instances.is_present
            FROM file_instances
            JOIN folders ON folders.id = file_instances.folder_id
            WHERE folders.absolute_path = ?
              AND file_instances.is_present = 1
            ORDER BY file_instances.absolute_path
            LIMIT ?
            """
        )
        defer { sqlite3_finalize(statement) }

        try bind(statement: statement, bindings: [.text(absolutePath), .integer(Int64(limit))])
        var records: [FileInstanceRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            records.append(
                FileInstanceRecord(
                    fileInstanceID: sqlite3_column_int64(statement, 0),
                    assetID: sqlite3_column_int64(statement, 1),
                    absolutePath: requireText(statement, at: 2),
                    relativePath: requireText(statement, at: 3),
                    isPresent: sqlite3_column_int64(statement, 4) != 0
                )
            )
        }
        return records
    }

    public func searchFileInstances(
        inFolderAbsolutePath absolutePath: String,
        query: String,
        limit: Int
    ) throws -> [FileInstanceRecord] {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedQuery.isEmpty == false, limit > 0 else {
            return []
        }

        let statement = try prepare(
            """
            SELECT file_instances.id, file_instances.asset_id, file_instances.absolute_path, file_instances.relative_path, file_instances.is_present
            FROM asset_terms
            JOIN file_instances ON file_instances.id = asset_terms.rowid
            JOIN folders ON folders.id = file_instances.folder_id
            WHERE folders.absolute_path = ?
              AND file_instances.is_present = 1
              AND asset_terms MATCH ?
            ORDER BY rank
            LIMIT ?
            """
        )
        defer { sqlite3_finalize(statement) }

        try bind(
            statement: statement,
            bindings: [
                .text(absolutePath),
                .text(matchQuery(for: trimmedQuery)),
                .integer(Int64(limit)),
            ]
        )

        var records: [FileInstanceRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            records.append(
                FileInstanceRecord(
                    fileInstanceID: sqlite3_column_int64(statement, 0),
                    assetID: sqlite3_column_int64(statement, 1),
                    absolutePath: requireText(statement, at: 2),
                    relativePath: requireText(statement, at: 3),
                    isPresent: sqlite3_column_int64(statement, 4) != 0
                )
            )
        }
        if records.isEmpty == false {
            return records
        }

        let substringStatement = try prepare(
            """
            SELECT file_instances.id, file_instances.asset_id, file_instances.absolute_path, file_instances.relative_path, file_instances.is_present
            FROM file_instances
            JOIN folders ON folders.id = file_instances.folder_id
            WHERE folders.absolute_path = ?
              AND file_instances.is_present = 1
              AND (
                lower(file_instances.relative_path) LIKE ? ESCAPE '\\'
                OR lower(file_instances.absolute_path) LIKE ? ESCAPE '\\'
              )
            ORDER BY
              CASE
                WHEN lower(file_instances.relative_path) = ? THEN 0
                WHEN lower(file_instances.relative_path) LIKE ? ESCAPE '\\' THEN 1
                ELSE 2
              END,
              length(file_instances.relative_path),
              file_instances.absolute_path
            LIMIT ?
            """
        )
        defer { sqlite3_finalize(substringStatement) }

        let normalizedQuery = trimmedQuery.lowercased()
        let escapedQuery = escapedLikeQuery(normalizedQuery)
        try bind(
            statement: substringStatement,
            bindings: [
                .text(absolutePath),
                .text("%\(escapedQuery)%"),
                .text("%\(escapedQuery)%"),
                .text(normalizedQuery),
                .text("\(escapedQuery)%"),
                .integer(Int64(limit)),
            ]
        )

        var substringRecords: [FileInstanceRecord] = []
        while sqlite3_step(substringStatement) == SQLITE_ROW {
            substringRecords.append(
                FileInstanceRecord(
                    fileInstanceID: sqlite3_column_int64(substringStatement, 0),
                    assetID: sqlite3_column_int64(substringStatement, 1),
                    absolutePath: requireText(substringStatement, at: 2),
                    relativePath: requireText(substringStatement, at: 3),
                    isPresent: sqlite3_column_int64(substringStatement, 4) != 0
                )
            )
        }

        return substringRecords
    }

    public func upsertEmbedding(
        assetID: Int64,
        encoderVersion: String,
        vector: [Double]
    ) throws {
        let blob = try encode(vector: vector)
        try execute(
            """
            INSERT INTO embeddings (asset_id, encoder_version, vector_blob, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(asset_id, encoder_version) DO UPDATE SET
              vector_blob = excluded.vector_blob,
              created_at = CURRENT_TIMESTAMP
            """,
            bindings: [
                .integer(assetID),
                .text(encoderVersion),
                .blob(blob),
            ]
        )
    }

    public func embeddingVector(
        assetID: Int64,
        encoderVersion: String
    ) throws -> [Double]? {
        let statement = try prepare(
            """
            SELECT vector_blob
            FROM embeddings
            WHERE asset_id = ?
              AND encoder_version = ?
            """
        )
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: [.integer(assetID), .text(encoderVersion)])
        guard sqlite3_step(statement) == SQLITE_ROW else {
            return nil
        }
        let blob = try requireBlob(statement, at: 0)
        return try decodeVector(blob)
    }

    public func missingEmbeddingWork(
        inFolderAbsolutePath absolutePath: String,
        encoderVersion: String
    ) throws -> [EmbeddingWorkRecord] {
        let statement = try prepare(
            """
            SELECT
              file_instances.asset_id,
              MIN(file_instances.absolute_path),
              MIN(file_instances.relative_path)
            FROM file_instances
            JOIN folders ON folders.id = file_instances.folder_id
            LEFT JOIN embeddings
              ON embeddings.asset_id = file_instances.asset_id
             AND embeddings.encoder_version = ?
            WHERE folders.absolute_path = ?
              AND file_instances.is_present = 1
              AND embeddings.id IS NULL
            GROUP BY file_instances.asset_id
            ORDER BY MIN(file_instances.absolute_path)
            """
        )
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: [.text(encoderVersion), .text(absolutePath)])

        var records: [EmbeddingWorkRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            records.append(
                EmbeddingWorkRecord(
                    assetID: sqlite3_column_int64(statement, 0),
                    representativeAbsolutePath: requireText(statement, at: 1),
                    representativeRelativePath: requireText(statement, at: 2)
                )
            )
        }
        return records
    }

    public func embeddedFiles(
        inFolderAbsolutePath absolutePath: String,
        encoderVersion: String
    ) throws -> [EmbeddedFileRecord] {
        let statement = try prepare(
            """
            SELECT
              file_instances.id,
              file_instances.asset_id,
              file_instances.absolute_path,
              file_instances.relative_path,
              embeddings.vector_blob
            FROM file_instances
            JOIN folders ON folders.id = file_instances.folder_id
            JOIN embeddings
              ON embeddings.asset_id = file_instances.asset_id
             AND embeddings.encoder_version = ?
            WHERE folders.absolute_path = ?
              AND file_instances.is_present = 1
            ORDER BY file_instances.absolute_path
            """
        )
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: [.text(encoderVersion), .text(absolutePath)])

        var records: [EmbeddedFileRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            records.append(
                EmbeddedFileRecord(
                    fileInstanceID: sqlite3_column_int64(statement, 0),
                    assetID: sqlite3_column_int64(statement, 1),
                    absolutePath: requireText(statement, at: 2),
                    relativePath: requireText(statement, at: 3),
                    vector: try decodeVector(requireBlob(statement, at: 4))
                )
            )
        }
        return records
    }

    public func embeddedFileCount(
        inFolderAbsolutePath absolutePath: String,
        encoderVersion: String
    ) throws -> Int {
        let statement = try prepare(
            """
            SELECT COUNT(*)
            FROM file_instances
            JOIN folders ON folders.id = file_instances.folder_id
            JOIN embeddings
              ON embeddings.asset_id = file_instances.asset_id
             AND embeddings.encoder_version = ?
            WHERE folders.absolute_path = ?
              AND file_instances.is_present = 1
            """
        )
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: [.text(encoderVersion), .text(absolutePath)])

        guard sqlite3_step(statement) == SQLITE_ROW else {
            return 0
        }
        return Int(sqlite3_column_int64(statement, 0))
    }

    public func markFileInstancesMissing(absolutePaths: [String]) throws {
        try removeFileInstances(absolutePaths: absolutePaths)
    }

    public func removeFileInstances(absolutePaths: [String]) throws {
        for absolutePath in absolutePaths {
            try execute(
                """
                DELETE FROM asset_terms
                WHERE rowid IN (
                  SELECT id
                  FROM file_instances
                  WHERE absolute_path = ?
                )
                """,
                bindings: [.text(absolutePath)]
            )

            try execute(
                """
                DELETE FROM file_instances
                WHERE absolute_path = ?
                """,
                bindings: [.text(absolutePath)]
            )
        }

        try purgeOrphanedAssets()
    }

    public func markMissingFileInstances(
        inFolderAbsolutePath absolutePath: String,
        keepingAbsolutePaths: Set<String>
    ) throws {
        let existingRows = try fileInstances(inFolderAbsolutePath: absolutePath)
        let missingPaths = existingRows
            .map(\.absolutePath)
            .filter { keepingAbsolutePaths.contains($0) == false }
        try removeFileInstances(absolutePaths: missingPaths)
    }

    public func insertTrainingRun(
        folderID: Int64,
        encoderVersion: String,
        status: String,
        summaryJSON: String?
    ) throws -> Int64 {
        try execute(
            """
            INSERT INTO training_runs (folder_id, encoder_version, status, started_at, summary_json)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            bindings: [
                .integer(folderID),
                .text(encoderVersion),
                .text(status),
                .text(summaryJSON ?? ""),
            ]
        )

        return sqlite3_last_insert_rowid(handle)
    }

    public func finishTrainingRun(
        id: Int64,
        status: String,
        summaryJSON: String?
    ) throws {
        try execute(
            """
            UPDATE training_runs
            SET status = ?,
                finished_at = CURRENT_TIMESTAMP,
                summary_json = ?
            WHERE id = ?
            """,
            bindings: [
                .text(status),
                .text(summaryJSON ?? ""),
                .integer(id),
            ]
        )
    }

    public func trainingRuns(inFolderAbsolutePath absolutePath: String) throws -> [TrainingRunRecord] {
        let statement = try prepare(
            """
            SELECT
              training_runs.id,
              training_runs.folder_id,
              training_runs.encoder_version,
              training_runs.status,
              training_runs.started_at,
              training_runs.finished_at,
              training_runs.summary_json
            FROM training_runs
            JOIN folders ON folders.id = training_runs.folder_id
            WHERE folders.absolute_path = ?
            ORDER BY training_runs.id DESC
            """
        )
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: [.text(absolutePath)])

        var records: [TrainingRunRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            records.append(
                TrainingRunRecord(
                    id: sqlite3_column_int64(statement, 0),
                    folderID: sqlite3_column_int64(statement, 1),
                    encoderVersion: requireText(statement, at: 2),
                    status: requireText(statement, at: 3),
                    startedAt: optionalText(statement, at: 4),
                    finishedAt: optionalText(statement, at: 5),
                    summaryJSON: optionalText(statement, at: 6)
                )
            )
        }
        return records
    }

    private func executeScript(_ sql: String) throws {
        var errorMessage: UnsafeMutablePointer<Int8>?
        if sqlite3_exec(handle, sql, nil, nil, &errorMessage) != SQLITE_OK {
            let message = errorMessage.map { String(cString: $0) } ?? "Database script failed."
            sqlite3_free(errorMessage)
            throw DatabaseError(message)
        }
    }

    private static func configureConnection(_ handle: OpaquePointer) throws {
        try executeStaticScript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA busy_timeout = 5000;
            PRAGMA foreign_keys = ON;
            """,
            handle: handle
        )
    }

    private static func executeStaticScript(_ sql: String, handle: OpaquePointer) throws {
        var errorMessage: UnsafeMutablePointer<Int8>?
        if sqlite3_exec(handle, sql, nil, nil, &errorMessage) != SQLITE_OK {
            let message = errorMessage.map { String(cString: $0) } ?? "Unable to execute database script."
            sqlite3_free(errorMessage)
            throw DatabaseError(message)
        }
    }

    private func execute(_ sql: String, bindings: [Binding]) throws {
        let statement = try prepare(sql)
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: bindings)
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw DatabaseError(lastErrorMessage())
        }
    }

    private func queryStrings(_ sql: String) throws -> [String] {
        let statement = try prepare(sql)
        defer { sqlite3_finalize(statement) }

        var values: [String] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            values.append(requireText(statement, at: 0))
        }
        return values
    }

    private func queryRow(_ sql: String, bindings: [Binding]) throws -> OpaquePointer {
        let statement = try prepare(sql)
        try bind(statement: statement, bindings: bindings)
        guard sqlite3_step(statement) == SQLITE_ROW else {
            sqlite3_finalize(statement)
            throw DatabaseError("No row returned.")
        }
        return statement
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw DatabaseError(lastErrorMessage())
        }
        return statement
    }

    private func bind(statement: OpaquePointer, bindings: [Binding]) throws {
        for (index, binding) in bindings.enumerated() {
            let position = Int32(index + 1)
            let result: Int32
            switch binding {
            case let .integer(value):
                if let value {
                    result = sqlite3_bind_int64(statement, position, value)
                } else {
                    result = sqlite3_bind_null(statement, position)
                }
            case let .text(value):
                result = value.withCString { pointer in
                    sqlite3_bind_text(statement, position, pointer, -1, SQLITE_TRANSIENT)
                }
            case let .blob(data):
                if let data {
                    result = data.withUnsafeBytes { bytes in
                        sqlite3_bind_blob(statement, position, bytes.baseAddress, Int32(data.count), SQLITE_TRANSIENT)
                    }
                } else {
                    result = sqlite3_bind_null(statement, position)
                }
            }

            guard result == SQLITE_OK else {
                throw DatabaseError(lastErrorMessage())
            }
        }
    }

    private func requireInteger(_ statement: OpaquePointer, at index: Int32) throws -> Int64 {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else {
            throw DatabaseError("Missing integer value.")
        }
        return sqlite3_column_int64(statement, index)
    }

    private func requireBlob(_ statement: OpaquePointer, at index: Int32) throws -> Data {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else {
            throw DatabaseError("Missing blob value.")
        }
        guard let bytes = sqlite3_column_blob(statement, index) else {
            return Data()
        }
        let count = Int(sqlite3_column_bytes(statement, index))
        return Data(bytes: bytes, count: count)
    }

    private func requireText(_ statement: OpaquePointer, at index: Int32) -> String {
        String(cString: sqlite3_column_text(statement, index))
    }

    private func optionalText(_ statement: OpaquePointer, at index: Int32) -> String? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else {
            return nil
        }
        return String(cString: sqlite3_column_text(statement, index))
    }

    private func encode(vector: [Double]) throws -> Data {
        guard vector.isEmpty == false else {
            throw DatabaseError("Embedding vector must not be empty.")
        }
        let values = vector.map(Float32.init)
        return values.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }

    private func decodeVector(_ data: Data) throws -> [Double] {
        guard data.count.isMultiple(of: MemoryLayout<Float32>.size) else {
            throw DatabaseError("Embedding blob has an invalid length.")
        }
        return data.withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: Float32.self)
            return values.map(Double.init)
        }
    }

    private func lastErrorMessage() -> String {
        sqlite3_errmsg(handle).map { String(cString: $0) } ?? "Database error."
    }

    private func matchQuery(for query: String) -> String {
        query
            .lowercased()
            .split(whereSeparator: \.isWhitespace)
            .map { token in
                let sanitized = token.filter { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
                return sanitized.isEmpty ? nil : "\(sanitized)*"
            }
            .compactMap { $0 }
            .joined(separator: " ")
    }

    private func escapedLikeQuery(_ query: String) -> String {
        var escaped = ""
        for character in query {
            switch character {
            case "\\", "%", "_":
                escaped.append("\\")
                escaped.append(character)
            default:
                escaped.append(character)
            }
        }
        return escaped
    }

    private func purgeOrphanedAssets() throws {
        try execute(
            """
            DELETE FROM assets
            WHERE id IN (
              SELECT assets.id
              FROM assets
              LEFT JOIN file_instances ON file_instances.asset_id = assets.id
              GROUP BY assets.id
              HAVING COUNT(file_instances.id) = 0
            )
            """,
            bindings: []
        )
    }
}

private enum Binding {
    case integer(Int64?)
    case text(String)
    case blob(Data?)
}

private struct DatabaseError: Error, LocalizedError {
    let errorDescription: String?

    init(_ message: String) {
        self.errorDescription = message
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
