import Dispatch
import Foundation
import OSLog

public enum SemanticGalleryRuntimeLogLevel: String, Sendable {
    case info
    case warning
    case error
}

public enum SemanticGalleryRuntimeLog {
    private static let queue = DispatchQueue(label: "SemanticGalleryRuntimeLog")
    private static let logger = Logger(subsystem: "com.semanticgallery.app", category: "runtime")

    public static func fileURL(paths: AppPaths) -> URL {
        paths.logsRoot.appending(path: "semanticgallery.log")
    }

    public static func record(
        _ message: String,
        level: SemanticGalleryRuntimeLogLevel = .info,
        category: String,
        paths: AppPaths,
        metadata: [String: String] = [:]
    ) {
        let logURL = fileURL(paths: paths)
        let line = formattedLine(
            message: message,
            level: level,
            category: category,
            metadata: metadata
        )
        recordSystemLog(line, level: level)

        queue.sync {
            let fileManager = FileManager.default
            do {
                try fileManager.createDirectory(
                    at: paths.logsRoot,
                    withIntermediateDirectories: true
                )

                let data = Data(line.utf8)
                if fileManager.fileExists(atPath: logURL.path(percentEncoded: false)) == false {
                    guard fileManager.createFile(atPath: logURL.path(percentEncoded: false), contents: nil) else {
                        throw CocoaError(.fileWriteUnknown)
                    }
                }
                let handle = try FileHandle(forWritingTo: logURL)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
            } catch {
                recordSystemLog(
                    "Runtime log file could not be updated. file=\(quoted(logURL.path(percentEncoded: false))) details=\(quoted(String(describing: error)))",
                    level: .error
                )
            }
        }
    }

    private static func recordSystemLog(
        _ message: String,
        level: SemanticGalleryRuntimeLogLevel
    ) {
        switch level {
        case .info:
            logger.info("\(message, privacy: .public)")
        case .warning:
            logger.warning("\(message, privacy: .public)")
        case .error:
            logger.error("\(message, privacy: .public)")
        }
    }

    private static func formattedLine(
        message: String,
        level: SemanticGalleryRuntimeLogLevel,
        category: String,
        metadata: [String: String]
    ) -> String {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let metadataText = metadata
            .filter { $0.value.isEmpty == false }
            .map { key, value in
                "\(key)=\(quoted(value.replacingOccurrences(of: "\n", with: " ")))"
            }
            .sorted()
            .joined(separator: " ")

        if metadataText.isEmpty {
            return "\(timestamp) [\(level.rawValue)] [\(category)] \(message)\n"
        }
        return "\(timestamp) [\(level.rawValue)] [\(category)] \(message) \(metadataText)\n"
    }

    private static func quoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        if escaped.contains(where: \.isWhitespace) {
            return "\"\(escaped)\""
        }
        return escaped
    }
}
