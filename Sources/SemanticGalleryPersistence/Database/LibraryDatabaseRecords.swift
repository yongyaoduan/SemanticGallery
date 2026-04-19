import Foundation

public struct FileInstanceRecord: Sendable, Equatable {
    public let fileInstanceID: Int64
    public let assetID: Int64
    public let absolutePath: String
    public let relativePath: String
    public let isPresent: Bool

    init(
        fileInstanceID: Int64,
        assetID: Int64,
        absolutePath: String,
        relativePath: String,
        isPresent: Bool
    ) {
        self.fileInstanceID = fileInstanceID
        self.assetID = assetID
        self.absolutePath = absolutePath
        self.relativePath = relativePath
        self.isPresent = isPresent
    }
}

public struct EmbeddingWorkRecord: Sendable, Equatable {
    public let assetID: Int64
    public let representativeAbsolutePath: String
    public let representativeRelativePath: String

    public init(assetID: Int64, representativeAbsolutePath: String, representativeRelativePath: String) {
        self.assetID = assetID
        self.representativeAbsolutePath = representativeAbsolutePath
        self.representativeRelativePath = representativeRelativePath
    }
}

public struct EmbeddedFileRecord: Sendable, Equatable {
    public let fileInstanceID: Int64
    public let assetID: Int64
    public let absolutePath: String
    public let relativePath: String
    public let vector: [Double]

    public init(
        fileInstanceID: Int64,
        assetID: Int64,
        absolutePath: String,
        relativePath: String,
        vector: [Double]
    ) {
        self.fileInstanceID = fileInstanceID
        self.assetID = assetID
        self.absolutePath = absolutePath
        self.relativePath = relativePath
        self.vector = vector
    }
}

public struct TrainingRunRecord: Sendable, Equatable {
    public let id: Int64
    public let folderID: Int64
    public let encoderVersion: String
    public let status: String
    public let startedAt: String?
    public let finishedAt: String?
    public let summaryJSON: String?

    init(
        id: Int64,
        folderID: Int64,
        encoderVersion: String,
        status: String,
        startedAt: String?,
        finishedAt: String?,
        summaryJSON: String?
    ) {
        self.id = id
        self.folderID = folderID
        self.encoderVersion = encoderVersion
        self.status = status
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.summaryJSON = summaryJSON
    }
}
