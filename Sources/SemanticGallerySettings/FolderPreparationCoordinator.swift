import Foundation
import SemanticGalleryCore
import SemanticGalleryPersistence

public actor FolderPreparationCoordinator {
    private let bookmarkStore: FolderBookmarkStore

    public init(bookmarkStore: FolderBookmarkStore = FolderBookmarkStore()) {
        self.bookmarkStore = bookmarkStore
    }

    public func prepareFolder(
        at url: URL,
        onProgress: (@Sendable (FolderPreparationProgress) async -> Void)? = nil
    ) async throws -> [FolderPreparationProgress] {
        var progressRows: [FolderPreparationProgress] = []

        let accessProgress = FolderPreparationProgress(
            step: .requestFolderAccess,
            message: "Authorizing the selected folder",
            progress: 0.25
        )
        progressRows.append(accessProgress)
        if let onProgress {
            await onProgress(accessProgress)
        }
        await Task.yield()

        let supportedCount = supportedImageCount(in: url)
        let scanProgress = FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Found \(supportedCount) supported images",
            progress: 0.5
        )
        progressRows.append(scanProgress)
        if let onProgress {
            await onProgress(scanProgress)
        }
        await Task.yield()

        try bookmarkStore.saveBookmark(for: url)
        let bookmarkProgress = FolderPreparationProgress(
            step: .persistBookmark,
            message: "Saved secure folder access",
            progress: 0.75
        )
        progressRows.append(bookmarkProgress)
        if let onProgress {
            await onProgress(bookmarkProgress)
        }
        await Task.yield()

        let finalizeProgress = FolderPreparationProgress(
            step: .finalizeFolderSelection,
            message: "Folder is ready",
            progress: 1.0
        )
        progressRows.append(finalizeProgress)
        if let onProgress {
            await onProgress(finalizeProgress)
        }
        await Task.yield()

        return progressRows
    }

    private func supportedImageCount(in url: URL) -> Int {
        let allowedExtensions = Set(["jpg", "jpeg", "png", "bmp", "tiff", "heic", "heif"])
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var count = 0
        for case let fileURL as URL in enumerator {
            if allowedExtensions.contains(fileURL.pathExtension.lowercased()) {
                count += 1
            }
        }
        return count
    }
}
