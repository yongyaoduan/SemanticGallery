import Foundation

public protocol GalleryEmbeddingService: Sendable {
    var encoderVersion: String { get }

    func encodeImage(at url: URL) async throws -> [Double]
    func encodeImages(at urls: [URL]) async throws -> [[Double]]
    func encodeImage(data: Data) async throws -> [Double]
    func encodeText(_ text: String) async throws -> [Double]
    func prepareForQueries() async throws
}

public extension GalleryEmbeddingService {
    func encodeImages(at urls: [URL]) async throws -> [[Double]] {
        var vectors: [[Double]] = []
        vectors.reserveCapacity(urls.count)
        for url in urls {
            vectors.append(try await encodeImage(at: url))
        }
        return vectors
    }

    func prepareForQueries() async throws {}
}
