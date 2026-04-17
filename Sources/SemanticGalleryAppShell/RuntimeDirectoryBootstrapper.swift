import Foundation
import SemanticGalleryPersistence

public enum RuntimeDirectoryBootstrapper {
    public static func ensureExists(
        for paths: AppPaths,
        fileManager: FileManager = .default
    ) {
        try? fileManager.createDirectory(
            at: paths.supportRoot,
            withIntermediateDirectories: true
        )
        try? fileManager.createDirectory(
            at: paths.cachesRoot,
            withIntermediateDirectories: true
        )
        try? fileManager.createDirectory(
            at: paths.logsRoot,
            withIntermediateDirectories: true
        )
    }
}
