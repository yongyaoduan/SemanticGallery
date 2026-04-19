import Foundation
import SQLite3

extension LibraryDatabase {
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

    public func missingEmbeddingWork(
        encoderVersion: String
    ) throws -> [EmbeddingWorkRecord] {
        let statement = try prepare(
            """
            SELECT
              file_instances.asset_id,
              MIN(file_instances.absolute_path),
              MIN(file_instances.relative_path)
            FROM file_instances
            LEFT JOIN embeddings
              ON embeddings.asset_id = file_instances.asset_id
             AND embeddings.encoder_version = ?
            WHERE file_instances.is_present = 1
              AND embeddings.id IS NULL
            GROUP BY file_instances.asset_id
            ORDER BY MIN(file_instances.absolute_path)
            """
        )
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: [.text(encoderVersion)])

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
}
