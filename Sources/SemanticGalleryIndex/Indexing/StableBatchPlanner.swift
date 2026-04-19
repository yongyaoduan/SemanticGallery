import CryptoKit
import Foundation

public struct StableBatchPlanner: Sendable {
    public let batchSize: Int
    public let seed: UInt64

    public init(batchSize: Int = 12, seed: UInt64 = 20_260_417) {
        self.batchSize = max(1, batchSize)
        self.seed = seed
    }

    public func makeBatches<Item>(
        _ items: [Item],
        stableKey: (Item) -> String
    ) -> [[Item]] {
        guard items.isEmpty == false else {
            return []
        }

        let orderedItems = items.sorted { lhs, rhs in
            let lhsKey = stableKey(lhs)
            let rhsKey = stableKey(rhs)
            let lhsDigest = digest(for: lhsKey)
            let rhsDigest = digest(for: rhsKey)

            if lhsDigest == rhsDigest {
                return lhsKey < rhsKey
            }
            return lhsDigest.lexicographicallyPrecedes(rhsDigest)
        }

        var batches: [[Item]] = []
        batches.reserveCapacity((orderedItems.count + batchSize - 1) / batchSize)

        var startIndex = 0
        while startIndex < orderedItems.count {
            let endIndex = min(startIndex + batchSize, orderedItems.count)
            batches.append(Array(orderedItems[startIndex..<endIndex]))
            startIndex = endIndex
        }

        return batches
    }

    private func digest(for key: String) -> [UInt8] {
        let input = "\(seed):\(key)"
        return Array(SHA256.hash(data: Data(input.utf8)))
    }
}
