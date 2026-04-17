import Foundation

public enum ArtifactRepositoryType: Equatable, Sendable {
    case model
    case dataset
}

public struct RemoteArtifact: Equatable, Sendable {
    public let repositoryID: String
    public let repositoryType: ArtifactRepositoryType
    public let relativePath: String
    public let requiredFiles: [String]

    public init(
        repositoryID: String,
        repositoryType: ArtifactRepositoryType = .model,
        relativePath: String,
        requiredFiles: [String]
    ) {
        self.repositoryID = repositoryID
        self.repositoryType = repositoryType
        self.relativePath = relativePath
        self.requiredFiles = requiredFiles
    }
}

public struct ArtifactCatalog: Equatable, Sendable {
    public let baseModel: RemoteArtifact
    public let stage1Checkpoint: RemoteArtifact
    public let publicAnchor: RemoteArtifact

    public init(
        baseModel: RemoteArtifact,
        stage1Checkpoint: RemoteArtifact,
        publicAnchor: RemoteArtifact
    ) {
        self.baseModel = baseModel
        self.stage1Checkpoint = stage1Checkpoint
        self.publicAnchor = publicAnchor
    }
}
