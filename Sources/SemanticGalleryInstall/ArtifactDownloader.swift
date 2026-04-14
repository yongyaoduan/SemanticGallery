import Foundation

public enum ArtifactDownloadError: Error, Equatable {
    case missingSource(String)
}

public protocol ArtifactDownloading: Sendable {
    func download(artifact: RemoteArtifact, into root: URL) async throws
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

    public func download(artifact: RemoteArtifact, into root: URL) async throws {
        let artifactRoot = root.appending(path: artifact.relativePath)
        try FileManager.default.createDirectory(at: artifactRoot, withIntermediateDirectories: true)

        switch mode {
        case .stub:
            try writeStubFiles(for: artifact, into: artifactRoot)
        case .live:
            try await copyLiveFiles(for: artifact, into: artifactRoot)
        }
    }

    private func writeStubFiles(for artifact: RemoteArtifact, into artifactRoot: URL) throws {
        for filename in artifact.requiredFiles {
            try Data(filename.utf8).write(to: artifactRoot.appending(path: filename))
        }
    }

    private func copyLiveFiles(for artifact: RemoteArtifact, into artifactRoot: URL) async throws {
        guard let sources = fileSources[artifact.relativePath] else {
            throw ArtifactDownloadError.missingSource(artifact.relativePath)
        }

        for filename in artifact.requiredFiles {
            guard let sourceURL = sources[filename] else {
                throw ArtifactDownloadError.missingSource("\(artifact.relativePath)/\(filename)")
            }

            let destinationURL = artifactRoot.appending(path: filename)

            if sourceURL.isFileURL {
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            } else {
                let (temporaryURL, _) = try await URLSession.shared.download(from: sourceURL)
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.moveItem(at: temporaryURL, to: destinationURL)
            }
        }
    }
}
