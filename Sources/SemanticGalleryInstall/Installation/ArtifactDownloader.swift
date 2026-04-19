import Foundation

public enum ArtifactDownloadError: Error, Equatable {
    case missingSource(String)
    case invalidRepositoryURL(String)
    case unexpectedResponse(String)
    case unexpectedStatusCode(Int, String)
}

public protocol ArtifactDownloading: Sendable {
    func download(
        artifact: RemoteArtifact,
        into root: URL,
        onProgress: (@Sendable (ArtifactTransferProgress) async -> Void)?
    ) async throws
}

public struct ArtifactTransferProgress: Equatable, Sendable {
    public let filename: String
    public let completedFileCount: Int
    public let totalFileCount: Int
    public let currentFileFraction: Double
    public let receivedBytes: Int64?
    public let expectedBytes: Int64?
    public let reusedExistingFile: Bool

    public init(
        filename: String,
        completedFileCount: Int,
        totalFileCount: Int,
        currentFileFraction: Double = 0,
        receivedBytes: Int64? = nil,
        expectedBytes: Int64? = nil,
        reusedExistingFile: Bool
    ) {
        self.filename = filename
        self.completedFileCount = completedFileCount
        self.totalFileCount = totalFileCount
        self.currentFileFraction = min(max(currentFileFraction, 0), 1)
        self.receivedBytes = receivedBytes
        self.expectedBytes = expectedBytes
        self.reusedExistingFile = reusedExistingFile
    }

    public var fractionalCompletedFileCount: Double {
        guard completedFileCount < totalFileCount else {
            return Double(totalFileCount)
        }
        return min(Double(totalFileCount), Double(completedFileCount) + currentFileFraction)
    }
}

public struct ArtifactDownloader: ArtifactDownloading, Sendable {
    public enum Mode: Sendable {
        case live
        case stub
    }

    private let mode: Mode
    private let fileSources: [String: [String: URL]]

    public static let stubbed = ArtifactDownloader(mode: .stub, fileSources: [:])

    public init(mode: Mode = .live, fileSources: [String: [String: URL]] = [:]) {
        self.mode = mode
        self.fileSources = fileSources
    }

    public func download(
        artifact: RemoteArtifact,
        into root: URL,
        onProgress: (@Sendable (ArtifactTransferProgress) async -> Void)? = nil
    ) async throws {
        let artifactRoot = root.appending(path: artifact.relativePath)
        try FileManager.default.createDirectory(at: artifactRoot, withIntermediateDirectories: true)
        try await copyFiles(for: artifact, into: artifactRoot, onProgress: onProgress)
    }

    func sourceURL(for artifact: RemoteArtifact, filename: String) throws -> URL {
        if let fileSource = fileSources[artifact.relativePath]?[filename] {
            return fileSource
        }

        let prefix: String
        switch artifact.repositoryType {
        case .model:
            prefix = "https://huggingface.co"
        case .dataset:
            prefix = "https://huggingface.co/datasets"
        }

        guard var components = URLComponents(string: "\(prefix)/\(artifact.repositoryID)/resolve/main/\(filename)") else {
            throw ArtifactDownloadError.invalidRepositoryURL("\(artifact.repositoryID)/\(filename)")
        }
        components.percentEncodedQuery = nil
        guard let url = components.url else {
            throw ArtifactDownloadError.invalidRepositoryURL("\(artifact.repositoryID)/\(filename)")
        }
        return url
    }

    private func copyFiles(
        for artifact: RemoteArtifact,
        into artifactRoot: URL,
        onProgress: (@Sendable (ArtifactTransferProgress) async -> Void)?
    ) async throws {
        switch mode {
        case .stub:
            try await writeStubFiles(for: artifact, into: artifactRoot, onProgress: onProgress)
        case .live:
            try await copyLiveFiles(for: artifact, into: artifactRoot, onProgress: onProgress)
        }
    }

    private func writeStubFiles(
        for artifact: RemoteArtifact,
        into artifactRoot: URL,
        onProgress: (@Sendable (ArtifactTransferProgress) async -> Void)?
    ) async throws {
        for (index, filename) in artifact.requiredFiles.enumerated() {
            let destinationURL = artifactRoot.appending(path: filename)
            let reusedExistingFile = shouldSkipDownload(at: destinationURL)
            if reusedExistingFile {
                if let onProgress {
                    await onProgress(
                        ArtifactTransferProgress(
                            filename: filename,
                            completedFileCount: index + 1,
                            totalFileCount: artifact.requiredFiles.count,
                            currentFileFraction: 0,
                            receivedBytes: nil,
                            expectedBytes: nil,
                            reusedExistingFile: true
                        )
                    )
                }
                continue
            }
            try stubContents(for: artifact, filename: filename).write(to: destinationURL)
            if let onProgress {
                await onProgress(
                        ArtifactTransferProgress(
                            filename: filename,
                            completedFileCount: index + 1,
                            totalFileCount: artifact.requiredFiles.count,
                            currentFileFraction: 0,
                            receivedBytes: nil,
                            expectedBytes: nil,
                            reusedExistingFile: reusedExistingFile
                        )
                    )
                }
        }
    }

    private func stubContents(for artifact: RemoteArtifact, filename: String) -> Data {
        if filename.hasSuffix(".json") {
            return stubJSONContents(for: artifact, filename: filename)
        }
        if filename == "weights.safetensors" {
            return stubSafetensorsContents()
        }
        if filename.hasSuffix(".tar.gz") {
            return Data([0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        }
        return Data(filename.utf8)
    }

    private func stubJSONContents(for artifact: RemoteArtifact, filename: String) -> Data {
        let payload: Any
        switch (artifact.relativePath, filename) {
        case ("mlx/siglip2-base-patch16-224-f32", "config.json"):
            payload = [
                "text_config": ["max_position_embeddings": 64],
                "vision_config": ["image_size": 224],
            ]
        case ("mlx/siglip2-base-patch16-224-f32", "tokenizer_config.json"):
            payload = [
                "max_length": 64,
                "pad_token": "[PAD]",
            ]
        case ("mlx/siglip2-base-patch16-224-f32", "special_tokens_map.json"):
            payload = [
                "pad_token": "[PAD]",
            ]
        case ("mlx/siglip2-base-patch16-224-f32", "preprocessor_config.json"):
            payload = [
                "size": 224,
                "do_resize": true,
            ]
        case ("mlx/siglip2-base-patch16-224-f32", "tokenizer.json"):
            payload = [
                "version": "1.0",
                "model": ["type": "WordPiece"],
            ]
        case ("semanticgallery/stage1", "summary.json"):
            payload = [
                "encoder_version": "stage1-stub",
                "source": "stub",
            ]
        case ("semanticgallery/stage2_public_anchor", "sample_info.json"):
            payload = [
                "count": 1,
                "items": [["filename": "sample.jpg"]],
            ]
        default:
            payload = ["stub": filename]
        }

        return (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data("{}".utf8)
    }

    private func stubSafetensorsContents() -> Data {
        let headerObject: [String: Any] = [
            "stub_tensor": [
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            ]
        ]
        let headerData = (try? JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])) ?? Data("{}".utf8)
        let headerLength = UInt64(headerData.count)
        let prefix = withUnsafeBytes(of: headerLength.littleEndian) { Data($0) }
        return prefix + headerData + Data([0x00, 0x00, 0x00, 0x00])
    }

    private func copyLiveFiles(
        for artifact: RemoteArtifact,
        into artifactRoot: URL,
        onProgress: (@Sendable (ArtifactTransferProgress) async -> Void)?
    ) async throws {
        for (index, filename) in artifact.requiredFiles.enumerated() {
            let destinationURL = artifactRoot.appending(path: filename)
            let reusedExistingFile = shouldSkipDownload(at: destinationURL)
            if reusedExistingFile {
                if let onProgress {
                    await onProgress(
                        ArtifactTransferProgress(
                            filename: filename,
                            completedFileCount: index + 1,
                            totalFileCount: artifact.requiredFiles.count,
                            currentFileFraction: 0,
                            receivedBytes: nil,
                            expectedBytes: nil,
                            reusedExistingFile: true
                        )
                    )
                }
                continue
            }
            let sourceURL = try sourceURL(for: artifact, filename: filename)

            if sourceURL.isFileURL {
                if let onProgress {
                    await onProgress(
                        ArtifactTransferProgress(
                            filename: filename,
                            completedFileCount: index,
                            totalFileCount: artifact.requiredFiles.count,
                            currentFileFraction: 0,
                            receivedBytes: nil,
                            expectedBytes: nil,
                            reusedExistingFile: false
                        )
                    )
                }
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            } else {
                let (temporaryURL, response) = try await downloadRemoteFile(from: sourceURL) { received, expected in
                    guard let onProgress else {
                        return
                    }
                    let fraction = expected > 0 ? Double(received) / Double(expected) : 0
                    await onProgress(
                        ArtifactTransferProgress(
                            filename: filename,
                            completedFileCount: index,
                            totalFileCount: artifact.requiredFiles.count,
                            currentFileFraction: fraction,
                            receivedBytes: received,
                            expectedBytes: expected > 0 ? expected : nil,
                            reusedExistingFile: false
                        )
                    )
                }
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw ArtifactDownloadError.unexpectedResponse(sourceURL.absoluteString)
                }
                guard (200 ..< 300).contains(httpResponse.statusCode) else {
                    throw ArtifactDownloadError.unexpectedStatusCode(httpResponse.statusCode, sourceURL.absoluteString)
                }
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.moveItem(at: temporaryURL, to: destinationURL)
            }
            if let onProgress {
                await onProgress(
                    ArtifactTransferProgress(
                        filename: filename,
                        completedFileCount: index + 1,
                        totalFileCount: artifact.requiredFiles.count,
                        currentFileFraction: 0,
                        receivedBytes: nil,
                        expectedBytes: nil,
                        reusedExistingFile: reusedExistingFile
                    )
                )
            }
        }
    }

    private func shouldSkipDownload(at url: URL) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = attributes[.size] as? NSNumber else {
            return false
        }
        return fileSize.int64Value > 0
    }

    private func downloadRemoteFile(
        from sourceURL: URL,
        onProgress: @escaping @Sendable (Int64, Int64) async -> Void
    ) async throws -> (URL, URLResponse) {
        let observer = RemoteDownloadObserver(onProgress: onProgress)
        return try await observer.download(from: sourceURL)
    }
}

private final class RemoteDownloadObserver: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Int64, Int64) async -> Void
    private var continuation: CheckedContinuation<(URL, URLResponse), Error>?
    private var temporaryURL: URL?
    private var fileHandle: FileHandle?
    private var response: URLResponse?
    private var totalBytesReceived: Int64 = 0
    private var totalBytesExpected: Int64 = NSURLSessionTransferSizeUnknown
    private var lastReportedBytes: Int64 = 0
    private var lastReportedAt: Date = .distantPast
    private var completionError: Error?

    init(onProgress: @escaping @Sendable (Int64, Int64) async -> Void) {
        self.onProgress = onProgress
    }

    func download(from sourceURL: URL) async throws -> (URL, URLResponse) {
        let persistentURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        if FileManager.default.createFile(atPath: persistentURL.path, contents: nil) == false {
            throw CocoaError(.fileWriteUnknown)
        }
        temporaryURL = persistentURL
        fileHandle = try FileHandle(forWritingTo: persistentURL)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        return try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            let task = session.dataTask(with: sourceURL)
            task.resume()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        self.response = response
        totalBytesExpected = response.expectedContentLength
        Task {
            await onProgress(0, totalBytesExpected)
        }
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        do {
            try fileHandle?.write(contentsOf: data)
        } catch {
            completionError = error
            dataTask.cancel()
            return
        }

        totalBytesReceived += Int64(data.count)
        let now = Date()
        let shouldReport = totalBytesReceived == totalBytesExpected
            || totalBytesReceived - lastReportedBytes >= 1_048_576
            || now.timeIntervalSince(lastReportedAt) >= 0.15

        guard shouldReport else {
            return
        }

        lastReportedBytes = totalBytesReceived
        lastReportedAt = now
        let expected = totalBytesExpected
        Task {
            await onProgress(totalBytesReceived, expected)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let continuation else {
            return
        }
        self.continuation = nil
        try? fileHandle?.close()
        fileHandle = nil

        if let completionError {
            cleanupTemporaryFile()
            continuation.resume(throwing: completionError)
            return
        }

        if let error {
            cleanupTemporaryFile()
            continuation.resume(throwing: error)
            return
        }

        guard let temporaryURL, let response else {
            cleanupTemporaryFile()
            continuation.resume(throwing: ArtifactDownloadError.unexpectedResponse(task.originalRequest?.url?.absoluteString ?? ""))
            return
        }

        if lastReportedBytes != totalBytesReceived {
            let received = totalBytesReceived
            let expected = totalBytesExpected
            Task {
                await onProgress(received, expected)
            }
        }
        continuation.resume(returning: (temporaryURL, response))
    }

    private func cleanupTemporaryFile() {
        if let temporaryURL, FileManager.default.fileExists(atPath: temporaryURL.path) {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        temporaryURL = nil
    }
}
