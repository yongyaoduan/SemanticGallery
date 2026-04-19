import CryptoKit
import Foundation
import SemanticGalleryPersistence

public struct FolderIndexSummary: Sendable, Equatable {
    public let folderID: Int64
    public let folderPath: String
    public let fileCount: Int
    public let uniqueAssetCount: Int

    public init(folderID: Int64, folderPath: String, fileCount: Int, uniqueAssetCount: Int) {
        self.folderID = folderID
        self.folderPath = folderPath
        self.fileCount = fileCount
        self.uniqueAssetCount = uniqueAssetCount
    }
}

public struct FolderIndexScanProgress: Sendable, Equatable {
    public let processedCount: Int
    public let totalCount: Int
    public let uniqueAssetCount: Int

    public init(processedCount: Int, totalCount: Int, uniqueAssetCount: Int) {
        self.processedCount = processedCount
        self.totalCount = totalCount
        self.uniqueAssetCount = uniqueAssetCount
    }
}

public struct FolderIndexer {
    private let database: LibraryDatabase

    public init(database: LibraryDatabase) {
        self.database = database
    }

    @discardableResult
    public func rebuildIndex(
        for folderURL: URL,
        onScanProgress: (@Sendable (FolderIndexScanProgress) async -> Void)? = nil
    ) async throws -> FolderIndexSummary {
        let folderPath = folderURL.path(percentEncoded: false)
        let folderID = try database.upsertFolder(
            absolutePath: folderPath,
            bookmarkData: nil,
            isActive: true
        )

        let fileURLs = supportedImageFiles(in: folderURL)
        var seenPaths = Set<String>()
        var uniqueAssetIDs = Set<Int64>()

        for (index, fileURL) in fileURLs.enumerated() {
            let absolutePath = fileURL.path(percentEncoded: false)
            seenPaths.insert(absolutePath)

            let fileData = try Data(contentsOf: fileURL)
            let assetID = try database.upsertAsset(
                contentHash: SHA256.hash(data: fileData).hexDigest,
                fileSize: Int64(fileData.count),
                pixelWidth: nil,
                pixelHeight: nil
            )
            uniqueAssetIDs.insert(assetID)

            let values = try fileURL.resourceValues(forKeys: [.contentModificationDateKey])
            let modificationNanoseconds = Int64((values.contentModificationDate ?? .distantPast).timeIntervalSince1970 * 1_000_000_000)

            try database.upsertFileInstance(
                assetID: assetID,
                folderID: folderID,
                absolutePath: absolutePath,
                relativePath: relativePath(for: fileURL, inside: folderURL),
                mtimeNanoseconds: modificationNanoseconds,
                isPresent: true
            )

            await onScanProgress?(
                FolderIndexScanProgress(
                    processedCount: index + 1,
                    totalCount: fileURLs.count,
                    uniqueAssetCount: uniqueAssetIDs.count
                )
            )
        }

        try database.markMissingFileInstances(
            inFolderAbsolutePath: folderPath,
            keepingAbsolutePaths: seenPaths
        )

        return FolderIndexSummary(
            folderID: folderID,
            folderPath: folderPath,
            fileCount: fileURLs.count,
            uniqueAssetCount: uniqueAssetIDs.count
        )
    }

    private func supportedImageFiles(in folderURL: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return enumerator.compactMap { candidate in
            guard let fileURL = candidate as? URL else {
                return nil
            }
            return SupportedImagePath(url: fileURL)?.url
        }
        .sorted { $0.path(percentEncoded: false) < $1.path(percentEncoded: false) }
    }

    private func relativePath(for fileURL: URL, inside folderURL: URL) -> String {
        let fileComponents = fileURL.standardizedFileURL.pathComponents
        let folderComponents = folderURL.standardizedFileURL.pathComponents

        if fileComponents.starts(with: folderComponents) {
            return fileComponents.dropFirst(folderComponents.count).joined(separator: "/")
        }

        let privatePrefixedFolderComponents: [String]
        if folderComponents.first == "/" {
            privatePrefixedFolderComponents = ["/", "private"] + folderComponents.dropFirst()
        } else {
            privatePrefixedFolderComponents = ["private"] + folderComponents
        }

        if fileComponents.starts(with: privatePrefixedFolderComponents) {
            return fileComponents.dropFirst(privatePrefixedFolderComponents.count).joined(separator: "/")
        }

        return fileURL.lastPathComponent
    }
}

private extension SHA256Digest {
    var hexDigest: String {
        map { String(format: "%02x", $0) }.joined()
    }
}
