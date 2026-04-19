import Foundation

public struct SearchAsset: Sendable, Equatable, Identifiable {
    public let fileInstanceID: Int64
    public let assetID: Int64
    public let absolutePath: String
    public let relativePath: String
    public let thumbnailPath: String?

    public var id: Int64 { fileInstanceID }
    public var filename: String { URL(filePath: absolutePath).lastPathComponent }

    public init(
        fileInstanceID: Int64? = nil,
        assetID: Int64,
        absolutePath: String,
        relativePath: String? = nil,
        thumbnailPath: String?
    ) {
        self.fileInstanceID = fileInstanceID ?? assetID
        self.assetID = assetID
        self.absolutePath = absolutePath
        self.relativePath = relativePath ?? URL(filePath: absolutePath).lastPathComponent
        self.thumbnailPath = thumbnailPath
    }
}
