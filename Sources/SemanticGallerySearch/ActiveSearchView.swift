import Foundation

public struct SearchAssetRecord: Sendable, Equatable, Identifiable {
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

public struct ActiveSearchView: Sendable, Equatable {
    public let items: [SearchAssetRecord]
    private let matrix: [[Float]]
    private let normalizedMatrix: [[Float]]

    public init(items: [SearchAssetRecord], matrix: [[Double]]) {
        self.items = items
        self.matrix = matrix.map { $0.map(Float.init) }
        self.normalizedMatrix = Self.normalize(rows: self.matrix)
    }

    public func search(queryVector: [Double], limit: Int) -> [SearchAssetRecord] {
        guard items.isEmpty == false, limit > 0 else {
            return []
        }

        let query = queryVector.map(Float.init)
        let magnitude = Self.magnitude(of: query)
        guard magnitude > 0 else {
            return []
        }

        let normalizedQuery = query.map { $0 / magnitude }
        let scored = items.enumerated().map { index, item in
            (item, dot(normalizedMatrix[index], normalizedQuery))
        }

        return scored
            .sorted { lhs, rhs in
                if lhs.1 == rhs.1 {
                    return lhs.0.absolutePath < rhs.0.absolutePath
                }
                return lhs.1 > rhs.1
            }
            .prefix(limit)
            .map(\.0)
    }

    public func searchSimilar(recordID: Int64, limit: Int) -> [SearchAssetRecord] {
        guard let index = items.firstIndex(where: { $0.id == recordID }) else {
            return []
        }

        return search(queryVector: matrix[index].map(Double.init), limit: limit + 1)
            .filter { $0.id != recordID }
            .prefix(limit)
            .map { $0 }
    }

    public func removing(recordIDs: Set<Int64>) -> ActiveSearchView {
        guard recordIDs.isEmpty == false else {
            return self
        }

        let remaining = items.enumerated().filter { _, item in
            recordIDs.contains(item.id) == false
        }

        return ActiveSearchView(
            items: remaining.map(\.element),
            matrix: remaining.map { index, _ in
                matrix[index].map(Double.init)
            }
        )
    }

    public func removing(assetIDs: Set<Int64>) -> ActiveSearchView {
        removing(recordIDs: assetIDs)
    }

    private static func normalize(rows: [[Float]]) -> [[Float]] {
        rows.map { row in
            let magnitude = magnitude(of: row)
            guard magnitude > 0 else {
                return Array(repeating: 0, count: row.count)
            }
            return row.map { $0 / magnitude }
        }
    }

    private static func magnitude(of row: [Float]) -> Float {
        sqrt(row.reduce(0) { partial, value in
            partial + value * value
        })
    }

    private func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { partial, pair in
            partial + pair.0 * pair.1
        }
    }
}
