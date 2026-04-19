import Foundation
import Testing
@testable import SemanticGallerySearch

/// Formal specification for callers:
/// Pre: `items = [a, b]`, `limit = 2`, and `queryVector ≠ 0`.
/// Post after `search(queryVector, limit)`:
/// the result is the `limit`-bounded permutation of `items`
/// sorted by descending cosine similarity, with `absolutePath` as the tie breaker.
@Test
func folderSearchIndexUsesExactCosineRanking() {
    let view = FolderSearchIndex(
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

/// Formal specification for callers:
/// Pre: `assetIDs = {2}`.
/// Post after `removing(assetIDs)`:
/// `items' = items \\ { item | item.assetID = 2 }`
/// and every later `search` is evaluated only over `items'`.
@Test
func folderSearchIndexCanDropDeletedPathsWithoutFullReload() {
    let view = FolderSearchIndex(
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

/// Formal specification for callers:
/// Pre: `items = [a0, a1, b]`, where `a0.assetID = a1.assetID`.
/// Post after `removing(resultIDs: {a0.id})`:
/// only the selected file-instance is removed, and every other visible instance remains searchable.
@Test
func folderSearchIndexRemovingResultIDsKeepsOtherInstancesOfTheSameAsset() {
    let view = FolderSearchIndex(
        items: [
            .init(fileInstanceID: 10, assetID: 1, absolutePath: "/tmp/library-a/shared.jpg", thumbnailPath: nil),
            .init(fileInstanceID: 11, assetID: 1, absolutePath: "/tmp/library-b/shared.jpg", thumbnailPath: nil),
            .init(fileInstanceID: 12, assetID: 2, absolutePath: "/tmp/library-b/other.jpg", thumbnailPath: nil),
        ],
        matrix: [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    let reduced = view.removing(resultIDs: [10])
    let matches = reduced.search(queryVector: [1.0, 0.0], limit: 10)

    #expect(reduced.items.map(\.id) == [11, 12])
    #expect(matches.map(\.absolutePath) == [
        "/tmp/library-b/shared.jpg",
        "/tmp/library-b/other.jpg",
    ])
}

/// Formal specification for callers:
/// Pre: `resultID = r ∈ ids(items)`.
/// Post after `searchSimilar(r, limit)`:
/// `r` is excluded from the result, and the remaining items are ranked by cosine similarity to `r`.
/// Pre': `resultID ∉ ids(items)`.
/// Post': the result is empty.
@Test
func folderSearchIndexSearchSimilarExcludesTheSourceAndRejectsUnknownIDs() {
    let view = FolderSearchIndex(
        items: [
            .init(fileInstanceID: 1, assetID: 1, absolutePath: "/tmp/library/source.jpg", thumbnailPath: nil),
            .init(fileInstanceID: 2, assetID: 2, absolutePath: "/tmp/library/near.jpg", thumbnailPath: nil),
            .init(fileInstanceID: 3, assetID: 3, absolutePath: "/tmp/library/far.jpg", thumbnailPath: nil),
        ],
        matrix: [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ]
    )

    let similar = view.searchSimilar(resultID: 1, limit: 2)
    let missing = view.searchSimilar(resultID: 99, limit: 2)

    #expect(similar.map(\.id) == [2, 3])
    #expect(similar.contains { $0.id == 1 } == false)
    #expect(missing.isEmpty)
}
