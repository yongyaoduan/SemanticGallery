import Foundation
import SQLite3

extension LibraryDatabase {
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
                .text(summaryJSON),
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
                .text(summaryJSON),
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
}
