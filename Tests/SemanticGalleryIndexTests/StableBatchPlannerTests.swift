import Foundation
import Testing
@testable import SemanticGalleryIndex

@Test
func stableBatchPlannerReturnsDeterministicShuffledBatches() {
    let items = [
        "Travel/01.jpg",
        "Travel/02.jpg",
        "Travel/03.jpg",
        "Travel/04.jpg",
        "Food/01.jpg",
        "Food/02.jpg",
        "Food/03.jpg",
        "Food/04.jpg",
    ]

    let planner = StableBatchPlanner(batchSize: 3, seed: 42)
    let first = planner.makeBatches(items, stableKey: { $0 })
    let second = planner.makeBatches(items, stableKey: { $0 })

    #expect(first == second)
    #expect(first.flatMap { $0 }.count == items.count)
    #expect(Set(first.flatMap { $0 }) == Set(items))
    #expect(first.flatMap { $0 } != items)
    #expect(first.allSatisfy { $0.count <= 3 })
}
