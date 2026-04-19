import Foundation

public struct FolderSearchIndex: Sendable, Equatable {
    public let items: [SearchAsset]
    private let matrix: [[Float]]
    private let normalizedMatrix: [[Float]]

    public init(items: [SearchAsset], matrix: [[Double]]) {
        self.items = items
        self.matrix = matrix.map { $0.map(Float.init) }
        self.normalizedMatrix = Self.normalize(rows: self.matrix)
    }

    public func search(queryVector: [Double], limit: Int) -> [SearchAsset] {
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

    public func searchSimilar(resultID: Int64, limit: Int) -> [SearchAsset] {
        guard let index = items.firstIndex(where: { $0.id == resultID }) else {
            return []
        }

        return search(queryVector: matrix[index].map(Double.init), limit: limit + 1)
            .filter { $0.id != resultID }
            .prefix(limit)
            .map { $0 }
    }

    public func removing(resultIDs: Set<Int64>) -> FolderSearchIndex {
        guard resultIDs.isEmpty == false else {
            return self
        }

        let remaining = items.enumerated().filter { _, item in
            resultIDs.contains(item.id) == false
        }

        return FolderSearchIndex(
            items: remaining.map(\.element),
            matrix: remaining.map { index, _ in
                matrix[index].map(Double.init)
            }
        )
    }

    public func removing(assetIDs: Set<Int64>) -> FolderSearchIndex {
        guard assetIDs.isEmpty == false else {
            return self
        }

        let remaining = items.enumerated().filter { _, item in
            assetIDs.contains(item.assetID) == false
        }

        return FolderSearchIndex(
            items: remaining.map(\.element),
            matrix: remaining.map { index, _ in
                matrix[index].map(Double.init)
            }
        )
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
