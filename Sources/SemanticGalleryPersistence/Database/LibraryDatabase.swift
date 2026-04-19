import Foundation
import SQLite3

public final class LibraryDatabase: @unchecked Sendable {
    let handle: OpaquePointer

    init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        sqlite3_close(handle)
    }

    func execute(_ sql: String, bindings: [Binding]) throws {
        let statement = try prepare(sql)
        defer { sqlite3_finalize(statement) }
        try bind(statement: statement, bindings: bindings)
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw DatabaseError(lastErrorMessage())
        }
    }

    func queryStrings(_ sql: String) throws -> [String] {
        let statement = try prepare(sql)
        defer { sqlite3_finalize(statement) }

        var values: [String] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            values.append(requireText(statement, at: 0))
        }
        return values
    }

    func queryRow(_ sql: String, bindings: [Binding]) throws -> OpaquePointer {
        let statement = try prepare(sql)
        try bind(statement: statement, bindings: bindings)
        guard sqlite3_step(statement) == SQLITE_ROW else {
            sqlite3_finalize(statement)
            throw DatabaseError("No row returned.")
        }
        return statement
    }

    func prepare(_ sql: String) throws -> OpaquePointer {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw DatabaseError(lastErrorMessage())
        }
        return statement
    }

    func bind(statement: OpaquePointer, bindings: [Binding]) throws {
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
                if let value {
                    result = value.withCString { pointer in
                        sqlite3_bind_text(statement, position, pointer, -1, SQLITE_TRANSIENT)
                    }
                } else {
                    result = sqlite3_bind_null(statement, position)
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

    func requireInteger(_ statement: OpaquePointer, at index: Int32) throws -> Int64 {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else {
            throw DatabaseError("Missing integer value.")
        }
        return sqlite3_column_int64(statement, index)
    }

    func requireBlob(_ statement: OpaquePointer, at index: Int32) throws -> Data {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else {
            throw DatabaseError("Missing blob value.")
        }
        guard let bytes = sqlite3_column_blob(statement, index) else {
            return Data()
        }
        let count = Int(sqlite3_column_bytes(statement, index))
        return Data(bytes: bytes, count: count)
    }

    func requireText(_ statement: OpaquePointer, at index: Int32) -> String {
        String(cString: sqlite3_column_text(statement, index))
    }

    func optionalText(_ statement: OpaquePointer, at index: Int32) -> String? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else {
            return nil
        }
        return String(cString: sqlite3_column_text(statement, index))
    }

    func encode(vector: [Double]) throws -> Data {
        guard vector.isEmpty == false else {
            throw DatabaseError("Embedding vector must not be empty.")
        }
        let values = vector.map(Float32.init)
        return values.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }

    func decodeVector(_ data: Data) throws -> [Double] {
        guard data.count.isMultiple(of: MemoryLayout<Float32>.size) else {
            throw DatabaseError("Embedding blob has an invalid length.")
        }
        return data.withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: Float32.self)
            return values.map(Double.init)
        }
    }

    func lastErrorMessage() -> String {
        sqlite3_errmsg(handle).map { String(cString: $0) } ?? "Database error."
    }

    func matchQuery(for query: String) -> String {
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

    func escapedLikeQuery(_ query: String) -> String {
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

    func purgeOrphanedAssets() throws {
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

enum Binding {
    case integer(Int64?)
    case text(String?)
    case blob(Data?)
}

struct DatabaseError: Error, LocalizedError {
    let errorDescription: String?

    init(_ message: String) {
        self.errorDescription = message
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
