import Foundation

public enum PrivateAdaptationEpochPlannerError: Error, Equatable {
    case insufficientPublicExamples(required: Int, actual: Int)
}

public struct PrivateAdaptationEpoch<PrivateItem: Sendable & Equatable, PublicItem: Sendable & Equatable>: Sendable, Equatable {
    public let privateItems: [PrivateItem]
    public let publicItems: [PublicItem]

    public init(privateItems: [PrivateItem], publicItems: [PublicItem]) {
        self.privateItems = privateItems
        self.publicItems = publicItems
    }
}

public struct PrivateAdaptationEpochPlanner: Sendable {
    public let epochCount: Int
    public let publicItemsPerEpoch: Int
    public let seed: UInt64

    public init(
        epochCount: Int = 10,
        publicItemsPerEpoch: Int = 100,
        seed: UInt64 = 20_260_417
    ) {
        self.epochCount = max(1, epochCount)
        self.publicItemsPerEpoch = max(1, publicItemsPerEpoch)
        self.seed = seed
    }

    public func makeEpochs<PrivateItem: Sendable & Equatable, PublicItem: Sendable & Equatable>(
        privateItems: [PrivateItem],
        publicItems: [PublicItem]
    ) throws -> [PrivateAdaptationEpoch<PrivateItem, PublicItem>] {
        let requiredPublicCount = epochCount * publicItemsPerEpoch
        guard publicItems.count >= requiredPublicCount else {
            throw PrivateAdaptationEpochPlannerError.insufficientPublicExamples(
                required: requiredPublicCount,
                actual: publicItems.count
            )
        }

        var generator = SeededGenerator(state: seed)
        var shuffledPublicItems = publicItems
        shuffledPublicItems.shuffle(using: &generator)

        return (0..<epochCount).map { epochIndex in
            let startIndex = epochIndex * publicItemsPerEpoch
            let endIndex = startIndex + publicItemsPerEpoch
            return PrivateAdaptationEpoch(
                privateItems: privateItems,
                publicItems: Array(shuffledPublicItems[startIndex..<endIndex])
            )
        }
    }
}

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = 6_364_136_223_846_793_005 &* state &+ 1
        return state
    }
}
