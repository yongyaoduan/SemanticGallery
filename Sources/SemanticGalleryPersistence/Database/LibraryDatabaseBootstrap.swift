import Foundation
import SQLite3

extension LibraryDatabase {
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
        try executeScript(libraryDatabaseSchemaSQL)
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

    func executeScript(_ sql: String) throws {
        var errorMessage: UnsafeMutablePointer<Int8>?
        if sqlite3_exec(handle, sql, nil, nil, &errorMessage) != SQLITE_OK {
            let message = errorMessage.map { String(cString: $0) } ?? "Database script failed."
            sqlite3_free(errorMessage)
            throw DatabaseError(message)
        }
    }

    static func configureConnection(_ handle: OpaquePointer) throws {
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

    static func executeStaticScript(_ sql: String, handle: OpaquePointer) throws {
        var errorMessage: UnsafeMutablePointer<Int8>?
        if sqlite3_exec(handle, sql, nil, nil, &errorMessage) != SQLITE_OK {
            let message = errorMessage.map { String(cString: $0) } ?? "Unable to execute database script."
            sqlite3_free(errorMessage)
            throw DatabaseError(message)
        }
    }
}

private let libraryDatabaseSchemaSQL =
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
