import Foundation
import Testing
@testable import SemanticGallerySearch

@Test
func activeSearchViewUsesExactCosineRanking() {
    let view = ActiveSearchView(
        items: [
            .init(assetID: 1, absolutePath: "/tmp/library/strong-x.jpg", thumbnailPath: nil),
            .init(assetID: 2, absolutePath: "/tmp/library/diagonal.jpg", thumbnailPath: nil),
        ],
        matrix: [
            [10.0, 0.0],
            [1.0, 1.0],
        ]
    )

    let matches = view.search(queryVector: [1.0, 1.0], limit: 2)
    #expect(matches.map(\.absolutePath) == ["/tmp/library/diagonal.jpg", "/tmp/library/strong-x.jpg"])
}

@Test
func activeSearchViewCanDropDeletedPathsWithoutFullReload() {
    let view = ActiveSearchView(
        items: [
            .init(assetID: 1, absolutePath: "/tmp/library/one.jpg", thumbnailPath: nil),
            .init(assetID: 2, absolutePath: "/tmp/library/two.jpg", thumbnailPath: nil),
            .init(assetID: 3, absolutePath: "/tmp/library/three.jpg", thumbnailPath: nil),
        ],
        matrix: [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    let reduced = view.removing(assetIDs: [2])
    let matches = reduced.search(queryVector: [0.0, 1.0], limit: 2)

    #expect(reduced.items.map(\.absolutePath) == ["/tmp/library/one.jpg", "/tmp/library/three.jpg"])
    #expect(matches.map(\.absolutePath) == ["/tmp/library/three.jpg", "/tmp/library/one.jpg"])
}
