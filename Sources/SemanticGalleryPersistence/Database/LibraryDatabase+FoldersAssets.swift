import Foundation
import SQLite3

extension LibraryDatabase {
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
}
