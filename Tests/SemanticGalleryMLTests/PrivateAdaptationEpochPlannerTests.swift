import Foundation
import Testing
@testable import SemanticGalleryML

@Test
func privateAdaptationEpochPlannerKeepsPrivateImagesFixedAndUsesPublicPoolWithoutReplacement() throws {
    let privateItems = (1...100).map { "private-\($0)" }
    let publicItems = (1...1000).map { "public-\($0)" }

    let planner = PrivateAdaptationEpochPlanner(
        epochCount: 10,
        publicItemsPerEpoch: 100,
        seed: 42
    )
    let epochs = try planner.makeEpochs(
        privateItems: privateItems,
        publicItems: publicItems
    )

    #expect(epochs.count == 10)
    #expect(epochs.allSatisfy { $0.privateItems == privateItems })
    #expect(epochs.allSatisfy { $0.publicItems.count == 100 })
    #expect(epochs.allSatisfy { Set($0.publicItems).count == 100 })
    #expect(Set(epochs.flatMap(\.publicItems)).count == 1000)
}

@Test
func privateAdaptationEpochPlannerRequiresEnoughPublicExamples() {
    let privateItems = (1...100).map { "private-\($0)" }
    let publicItems = (1...250).map { "public-\($0)" }

    let planner = PrivateAdaptationEpochPlanner(
        epochCount: 10,
        publicItemsPerEpoch: 100,
        seed: 42
    )

    #expect(throws: PrivateAdaptationEpochPlannerError.self) {
        try planner.makeEpochs(
            privateItems: privateItems,
            publicItems: publicItems
        )
    }
}
