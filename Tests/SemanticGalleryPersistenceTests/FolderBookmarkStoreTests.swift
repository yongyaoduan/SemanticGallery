import Foundation
import Testing
@testable import SemanticGalleryPersistence

@Suite("Folder Bookmark Store Contracts")
struct FolderBookmarkStoreTests {
    /// Formal specification for callers:
    /// Pre: a writable bookmark file path `b` and an existing folder URL `f`.
    /// Post after `saveBookmark(for: f)` followed by `loadBookmark()`:
    /// `loadBookmark()' = f`.
    /// Post after `clear()`:
    /// `loadBookmark()'' = nil`.
    @Test
    func fileBackedStoreRoundTripsOneSelectedFolderUntilItIsCleared() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let selectedFolder = root.appendingPathComponent("Selected", isDirectory: true)
        let bookmarkFile = root.appendingPathComponent("Support/selected-folder.bookmark")
        try FileManager.default.createDirectory(at: selectedFolder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let store = FolderBookmarkStore(bookmarkFileURL: bookmarkFile)
        try store.saveBookmark(for: selectedFolder)

        let loaded = try store.loadBookmark()
        #expect(loaded?.standardizedFileURL == selectedFolder.standardizedFileURL)
        #expect(FileManager.default.fileExists(atPath: bookmarkFile.path))

        store.clear()
        #expect(try store.loadBookmark() == nil)
        #expect(FileManager.default.fileExists(atPath: bookmarkFile.path) == false)
    }

    /// Formal specification for callers:
    /// Pre: a dedicated defaults suite `d` and an existing folder URL `f`.
    /// Post after `saveBookmark(for: f)` followed by `loadBookmark()`:
    /// the same suite can recover `f`.
    /// Post after `clear()`:
    /// the suite no longer stores a selected folder bookmark.
    @Test
    func defaultsBackedStoreUsesOnlyTheProvidedSuiteAndClearsItCleanly() throws {
        let suiteName = "SemanticGalleryTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let selectedFolder = root.appendingPathComponent("Selected", isDirectory: true)
        try FileManager.default.createDirectory(at: selectedFolder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let store = FolderBookmarkStore(defaults: defaults)
        try store.saveBookmark(for: selectedFolder)

        let loaded = try store.loadBookmark()
        #expect(loaded?.standardizedFileURL == selectedFolder.standardizedFileURL)

        store.clear()
        #expect(try store.loadBookmark() == nil)
    }
}
