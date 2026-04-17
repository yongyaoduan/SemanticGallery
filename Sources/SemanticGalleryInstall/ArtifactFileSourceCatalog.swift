import Foundation

public struct ArtifactFileSourceCatalog: Sendable {
    public let root: URL
    public let catalog: ArtifactCatalog

    public init(root: URL, catalog: ArtifactCatalog = .legacyCompatible) {
        self.root = root
        self.catalog = catalog
    }

    public var fileSources: [String: [String: URL]] {
        var sources: [String: [String: URL]] = [:]
        for artifact in [catalog.baseModel, catalog.stage1Checkpoint, catalog.publicAnchor] {
            sources[artifact.relativePath] = Dictionary(uniqueKeysWithValues: artifact.requiredFiles.map { filename in
                (
                    filename,
                    root
                        .appending(path: artifact.relativePath)
                        .appending(path: filename)
                )
            })
        }
        return sources
    }
}
