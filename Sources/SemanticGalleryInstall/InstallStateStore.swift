import Foundation
import SemanticGalleryPersistence

public struct InstallStateStore {
    public let paths: AppPaths
    public let catalog: ArtifactCatalog

    public init(paths: AppPaths, catalog: ArtifactCatalog = .legacyCompatible) {
        self.paths = paths
        self.catalog = catalog
    }

    public func isInstallationComplete() throws -> Bool {
        let fileManager = FileManager.default

        func artifactComplete(_ artifact: RemoteArtifact) -> Bool {
            let artifactRoot = paths.artifactsRoot.appending(path: artifact.relativePath)
            return artifact.requiredFiles.allSatisfy { filename in
                fileManager.fileExists(atPath: artifactRoot.appending(path: filename).path)
            }
        }

        guard artifactComplete(catalog.baseModel) else { return false }
        guard artifactComplete(catalog.stage1Checkpoint) else { return false }
        guard artifactComplete(catalog.publicAnchor) else { return false }
        return fileManager.fileExists(atPath: paths.installStateURL.path)
    }
}
