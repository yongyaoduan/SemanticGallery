import Foundation

public struct FolderBookmarkStore: @unchecked Sendable {
    private let defaults: UserDefaults?
    private let bookmarkFileURL: URL?
    private let bookmarkKey = "semanticgallery.selected-folder-bookmark"

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.bookmarkFileURL = nil
    }

    public init(bookmarkFileURL: URL) {
        self.defaults = nil
        self.bookmarkFileURL = bookmarkFileURL
    }

    public func saveBookmark(for url: URL) throws {
        let data = try url.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        if let bookmarkFileURL {
            try FileManager.default.createDirectory(
                at: bookmarkFileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: bookmarkFileURL, options: .atomic)
        } else {
            defaults?.set(data, forKey: bookmarkKey)
        }
    }

    public func loadBookmark() throws -> URL? {
        let data: Data?
        if let bookmarkFileURL {
            guard FileManager.default.fileExists(atPath: bookmarkFileURL.path) else {
                return nil
            }
            data = try Data(contentsOf: bookmarkFileURL)
        } else {
            data = defaults?.data(forKey: bookmarkKey)
        }

        guard let data else {
            return nil
        }
        var isStale = false
        return try URL(
            resolvingBookmarkData: data,
            options: .withSecurityScope,
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        )
    }

    public func clear() {
        if let bookmarkFileURL {
            try? FileManager.default.removeItem(at: bookmarkFileURL)
        } else {
            defaults?.removeObject(forKey: bookmarkKey)
        }
    }
}
