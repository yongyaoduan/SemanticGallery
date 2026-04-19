import Foundation

public struct SupportedImagePath: Sendable, Equatable {
    private static let supportedExtensions = Set([
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tiff",
        "heic",
        "heif",
    ])

    public let url: URL

    public init?(url: URL) {
        guard Self.supportedExtensions.contains(url.pathExtension.lowercased()) else {
            return nil
        }
        self.url = url
    }
}
