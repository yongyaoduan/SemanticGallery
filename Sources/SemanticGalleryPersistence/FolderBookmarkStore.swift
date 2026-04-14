import Foundation

public struct FolderBookmarkStore {
    private let defaults: UserDefaults
    private let bookmarkKey = "semanticgallery.selected-folder-bookmark"

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public func saveBookmark(for url: URL) throws {
        let data = try url.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        defaults.set(data, forKey: bookmarkKey)
    }

    public func loadBookmark() throws -> URL? {
        guard let data = defaults.data(forKey: bookmarkKey) else {
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
        defaults.removeObject(forKey: bookmarkKey)
    }
}
