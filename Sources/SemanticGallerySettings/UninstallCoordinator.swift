import Foundation
import SemanticGalleryPersistence

public struct UninstallCoordinator {
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func removeArtifacts(paths: AppPaths, selectedFolder: URL?) throws {
        let ownedURLs = [paths.supportRoot, paths.cachesRoot, paths.logsRoot]

        for url in ownedURLs where url != selectedFolder {
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
        }

        FolderBookmarkStore().clear()
    }
}
