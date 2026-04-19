import CryptoKit
import Foundation
import SemanticGalleryCore
import SemanticGalleryPersistence

public struct FolderAdaptationArtifact: Sendable, Equatable, Codable {
    public let folderKey: String
    public let folderPath: String
    public let encoderVersion: String
    public let adapterWeightsURL: URL
    public let summaryURL: URL
    public let trainedImageCount: Int

    public init(
        folderKey: String,
        folderPath: String,
        encoderVersion: String,
        adapterWeightsURL: URL,
        summaryURL: URL,
        trainedImageCount: Int
    ) {
        self.folderKey = folderKey
        self.folderPath = folderPath
        self.encoderVersion = encoderVersion
        self.adapterWeightsURL = adapterWeightsURL
        self.summaryURL = summaryURL
        self.trainedImageCount = trainedImageCount
    }
}

public struct GalleryAdaptationResult: Sendable {
    public let artifact: FolderAdaptationArtifact
    public let embeddingService: any GalleryEmbeddingService

    public init(artifact: FolderAdaptationArtifact, embeddingService: any GalleryEmbeddingService) {
        self.artifact = artifact
        self.embeddingService = embeddingService
    }
}

public protocol GalleryAdaptationTraining: Sendable {
    func runAdaptation(
        for folderURL: URL,
        progress: @escaping @Sendable (PrivateAdaptationProgress) async -> Void
    ) async throws -> GalleryAdaptationResult

    func existingArtifact(for folderURL: URL) async throws -> GalleryAdaptationResult?
}

public struct FolderAdaptationArtifactStore: Sendable {
    private let paths: AppPaths

    public init(paths: AppPaths) {
        self.paths = paths
    }

    public func save(_ artifact: FolderAdaptationArtifact) throws {
        let metadataURL = metadataURL(forFolderKey: artifact.folderKey)
        try FileManager.default.createDirectory(at: metadataURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(artifact)
        try data.write(to: metadataURL, options: .atomic)
    }

    public func artifact(for folderURL: URL) -> FolderAdaptationArtifact? {
        let folderKey = Self.folderKey(for: folderURL)
        let metadataURL = metadataURL(forFolderKey: folderKey)
        guard let data = try? Data(contentsOf: metadataURL) else {
            return nil
        }
        guard let artifact = try? JSONDecoder().decode(FolderAdaptationArtifact.self, from: data) else {
            return nil
        }
        guard artifact.folderPath == folderURL.path(percentEncoded: false) else {
            return nil
        }
        guard FileManager.default.fileExists(atPath: artifact.adapterWeightsURL.path(percentEncoded: false)) else {
            return nil
        }
        return artifact
    }

    public static func folderKey(for folderURL: URL) -> String {
        let resolved = folderURL.standardizedFileURL.path(percentEncoded: false)
        let digest = SHA256.hash(data: Data(resolved.utf8))
        let prefix = digest.prefix(6).map { String(format: "%02x", $0) }.joined()
        let slug = folderURL.lastPathComponent
            .lowercased()
            .map { character -> Character in
                if character.isLetter || character.isNumber {
                    return character
                }
                return "-"
            }
        let slugString = String(slug).split(separator: "-").joined(separator: "-")
        return slugString.isEmpty ? prefix : "\(slugString)-\(prefix)"
    }

    public func baseDirectory(for folderURL: URL) -> URL {
        paths.modelsRoot
            .appending(path: "Adapted")
            .appending(path: Self.folderKey(for: folderURL))
    }

    private func metadataURL(forFolderKey folderKey: String) -> URL {
        paths.modelsRoot
            .appending(path: "Adapted")
            .appending(path: folderKey)
            .appending(path: "metadata.json")
    }
}
