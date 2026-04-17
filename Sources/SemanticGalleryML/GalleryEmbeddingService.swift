import Foundation

public protocol GalleryEmbeddingService: Sendable {
    var encoderVersion: String { get }

    func encodeImage(at url: URL) async throws -> [Double]
    func encodeImage(data: Data) async throws -> [Double]
    func encodeText(_ text: String) async throws -> [Double]
    func prepareForQueries() async throws
}

public extension GalleryEmbeddingService {
    func prepareForQueries() async throws {}
}
