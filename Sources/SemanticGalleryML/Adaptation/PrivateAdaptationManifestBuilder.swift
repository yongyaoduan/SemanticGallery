import Foundation
import SemanticGalleryIndex

public struct PrivateAdaptationImage: Sendable, Equatable, Codable {
    public let absolutePath: String
    public let relativePath: String
    public let sourceGroup: String

    public init(absolutePath: String, relativePath: String, sourceGroup: String) {
        self.absolutePath = absolutePath
        self.relativePath = relativePath
        self.sourceGroup = sourceGroup
    }
}

public struct PrivateAdaptationManifestBuilder: Sendable {
    public let maximumImageCount: Int
    public let seed: UInt64

    public init(maximumImageCount: Int = 100, seed: UInt64 = 20_260_403) {
        self.maximumImageCount = maximumImageCount
        self.seed = seed
    }

    public func makeManifest(from folderURL: URL) throws -> [PrivateAdaptationImage] {
        let fileURLs = supportedImageFiles(in: folderURL)
        guard fileURLs.isEmpty == false else {
            return []
        }

        var grouped: [String: [PrivateAdaptationImage]] = [:]
        for fileURL in fileURLs {
            let group = sourceGroup(for: fileURL, root: folderURL)
            grouped[group, default: []].append(
                PrivateAdaptationImage(
                    absolutePath: fileURL.path(percentEncoded: false),
                    relativePath: relativePath(for: fileURL, inside: folderURL),
                    sourceGroup: group
                )
            )
        }

        let generatorSeed = Int(truncatingIfNeeded: seed)
        var generator = SeededGenerator(state: UInt64(bitPattern: Int64(generatorSeed)))
        let sortedGroups = grouped.keys.sorted()
        for group in sortedGroups {
            grouped[group]?.shuffle(using: &generator)
        }

        var selected: [PrivateAdaptationImage] = []
        let targetCount = min(maximumImageCount, fileURLs.count)

        while selected.count < targetCount {
            var madeProgress = false
            for group in sortedGroups {
                guard var items = grouped[group], items.isEmpty == false else {
                    continue
                }
                selected.append(items.removeFirst())
                grouped[group] = items
                madeProgress = true
                if selected.count == targetCount {
                    break
                }
            }
            if madeProgress == false {
                break
            }
        }

        return selected
    }

    private func supportedImageFiles(in folderURL: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return enumerator.compactMap { candidate in
            guard let fileURL = candidate as? URL else {
                return nil
            }
            return SupportedImagePath(url: fileURL)?.url
        }
        .sorted { $0.path(percentEncoded: false) < $1.path(percentEncoded: false) }
    }

    private func relativePath(for fileURL: URL, inside folderURL: URL) -> String {
        let fileComponents = fileURL.standardizedFileURL.pathComponents
        let folderComponents = folderURL.standardizedFileURL.pathComponents

        if fileComponents.starts(with: folderComponents) {
            return fileComponents.dropFirst(folderComponents.count).joined(separator: "/")
        }

        return fileURL.lastPathComponent
    }

    private func sourceGroup(for fileURL: URL, root folderURL: URL) -> String {
        let relativePath = relativePath(for: fileURL, inside: folderURL)
        if let firstComponent = relativePath.split(separator: "/").first, relativePath.contains("/") {
            return String(firstComponent)
        }
        return folderURL.lastPathComponent
    }
}

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = 6_364_136_223_846_793_005 &* state &+ 1
        return state
    }
}
