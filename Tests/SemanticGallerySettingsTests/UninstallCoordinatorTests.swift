import Foundation
import Testing
@testable import SemanticGalleryPersistence
@testable import SemanticGallerySettings

@Test
func uninstallRemovesAppArtifactsButKeepsUserFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let userFolder = root.appendingPathComponent("UserAlbum")
    try FileManager.default.createDirectory(at: userFolder, withIntermediateDirectories: true)
    let suiteName = "SemanticGalleryUninstallTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { try? FileManager.default.removeItem(at: root) }
    defer { defaults.removePersistentDomain(forName: suiteName) }

    let paths = AppPaths(root: root)
    try FileManager.default.createDirectory(at: paths.supportRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: paths.cachesRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: paths.logsRoot, withIntermediateDirectories: true)

    let coordinator = UninstallCoordinator(
        bookmarkStore: FolderBookmarkStore(defaults: defaults),
        temporaryRoot: root
    )
    try coordinator.removeArtifacts(paths: paths, selectedFolder: userFolder)

    #expect(FileManager.default.fileExists(atPath: userFolder.path))
    #expect(FileManager.default.fileExists(atPath: paths.supportRoot.path) == false)
    #expect(FileManager.default.fileExists(atPath: paths.cachesRoot.path) == false)
    #expect(FileManager.default.fileExists(atPath: paths.logsRoot.path) == false)
}

@Test
func selfRemovalOnlyTargetsInstalledSemanticGalleryBundle() {
    let coordinator = UninstallCoordinator()
    let installedURL = URL(filePath: "/Applications/SemanticGallery.app")
    let buildURL = URL(filePath: "/tmp/Build/Products/Release/SemanticGallery.app")
    let wrongNameURL = URL(filePath: "/Applications/SemanticGallery Legacy.app")

    #expect(coordinator.shouldSelfRemove(appBundleURL: installedURL))
    #expect(coordinator.shouldSelfRemove(appBundleURL: buildURL) == false)
    #expect(coordinator.shouldSelfRemove(appBundleURL: wrongNameURL) == false)
}

@Test
func selfRemovalScriptClearsPreferencesAndPreferenceCacheAfterTheAppQuits() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let homeRoot = root.appendingPathComponent("Home")
    try FileManager.default.createDirectory(at: homeRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let coordinator = UninstallCoordinator(
        homeRoot: homeRoot,
        bundleIdentifier: "com.semanticgallery.app"
    )

    let scriptURL = try coordinator.writeSelfRemovalScript(
        appBundleURL: URL(filePath: "/Applications/SemanticGallery.app"),
        processIdentifier: 4242,
        scriptRoot: root
    )
    let script = try String(contentsOf: scriptURL, encoding: .utf8)

    #expect(script.contains("Library/Preferences/com.semanticgallery.app.plist"))
    #expect(script.contains("Library/Preferences/ByHost"))
    #expect(script.contains("Library/Saved Application State/com.semanticgallery.app.savedState"))
    #expect(script.contains("/usr/bin/defaults delete 'com.semanticgallery.app'"))
    #expect(script.contains("/usr/bin/defaults -currentHost delete 'com.semanticgallery.app'"))
    #expect(script.contains("/usr/bin/killall cfprefsd"))
}

@Test
func uninstallRemovesPreferencesSavedStateAndTemporaryResidue() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let homeRoot = root.appendingPathComponent("Home")
    let tempRoot = root.appendingPathComponent("Temp")
    try FileManager.default.createDirectory(at: homeRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let bundleIdentifier = "com.semanticgallery.app"
    let preferencesURL = homeRoot.appendingPathComponent("Library/Preferences/\(bundleIdentifier).plist")
    let byHostPreferencesURL = homeRoot.appendingPathComponent("Library/Preferences/ByHost/\(bundleIdentifier).host.plist")
    let savedStateURL = homeRoot.appendingPathComponent("Library/Saved Application State/\(bundleIdentifier).savedState")
    let tempResidueURL = tempRoot.appendingPathComponent("residual-root/SemanticGallery")
    try FileManager.default.createDirectory(at: preferencesURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: byHostPreferencesURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: savedStateURL, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tempResidueURL, withIntermediateDirectories: true)
    try Data("prefs".utf8).write(to: preferencesURL)
    try Data("prefs".utf8).write(to: byHostPreferencesURL)
    try Data("install".utf8).write(to: tempResidueURL.appendingPathComponent("install-state.json"))

    let suiteName = "SemanticGalleryUninstallTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defaults.set("folder", forKey: "semanticgallery.selected-folder-bookmark")
    defer { defaults.removePersistentDomain(forName: suiteName) }

    let coordinator = UninstallCoordinator(
        bookmarkStore: FolderBookmarkStore(defaults: defaults),
        homeRoot: homeRoot,
        temporaryRoot: tempRoot,
        bundleIdentifier: bundleIdentifier
    )

    try coordinator.removeArtifacts(paths: AppPaths(root: root.appendingPathComponent("Runtime")), selectedFolder: nil)

    #expect(FileManager.default.fileExists(atPath: preferencesURL.path) == false)
    #expect(FileManager.default.fileExists(atPath: byHostPreferencesURL.path) == false)
    #expect(FileManager.default.fileExists(atPath: savedStateURL.path) == false)
    #expect(FileManager.default.fileExists(atPath: tempResidueURL.path) == false)
    #expect(defaults.object(forKey: "semanticgallery.selected-folder-bookmark") == nil)
}
