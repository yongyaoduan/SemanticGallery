import Foundation
import Testing
@testable import SemanticGalleryML

@Test
func privateAdaptationManifestBuilderCapsAtOneHundredRowsAndBalancesParentFolders() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let travel = root.appending(path: "Travel")
    let food = root.appending(path: "Food")
    let notes = root.appending(path: "Notes")
    try FileManager.default.createDirectory(at: travel, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: food, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: notes, withIntermediateDirectories: true)

    for index in 0..<60 {
        try Data("travel-\(index)".utf8).write(to: travel.appending(path: "travel-\(index).jpg"))
    }
    for index in 0..<60 {
        try Data("food-\(index)".utf8).write(to: food.appending(path: "food-\(index).jpg"))
    }
    for index in 0..<60 {
        try Data("notes-\(index)".utf8).write(to: notes.appending(path: "notes-\(index).jpg"))
    }

    let builder = PrivateAdaptationManifestBuilder(maximumImageCount: 100, seed: 42)
    let manifest = try builder.makeManifest(from: root)

    #expect(manifest.count == 100)

    let countsByGroup = Dictionary(grouping: manifest, by: \.sourceGroup)
        .mapValues(\.count)
    let minCount = countsByGroup.values.min()
    let maxCount = countsByGroup.values.max()

    #expect(countsByGroup.count == 3)
    #expect(minCount == 33)
    #expect(maxCount == 34)
}

@Test
func privateAdaptationManifestBuilderKeepsEveryImageWhenFolderIsBelowTheCap() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    for index in 0..<12 {
        try Data("image-\(index)".utf8).write(to: root.appending(path: "image-\(index).jpg"))
    }

    let builder = PrivateAdaptationManifestBuilder(maximumImageCount: 100, seed: 42)
    let manifest = try builder.makeManifest(from: root)

    #expect(manifest.count == 12)
    #expect(Set(manifest.map(\.relativePath)).count == 12)
}
