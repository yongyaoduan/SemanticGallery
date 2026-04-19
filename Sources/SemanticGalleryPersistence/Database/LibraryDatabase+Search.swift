import Foundation
import SQLite3

extension LibraryDatabase {
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
}
