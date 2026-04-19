import Foundation
import CoreFoundation
import SemanticGalleryPersistence

public struct UninstallCoordinator {
    private let selfRemovalDelaySeconds: Int
    private let fileManager: FileManager
    private let bookmarkStore: FolderBookmarkStore
    private let homeRoot: URL
    private let temporaryRoot: URL
    private let bundleIdentifier: String

    public init(
        fileManager: FileManager = .default,
        bookmarkStore: FolderBookmarkStore = FolderBookmarkStore(),
        selfRemovalDelaySeconds: Int = 3,
        homeRoot: URL = URL(filePath: NSHomeDirectory(), directoryHint: .isDirectory),
        temporaryRoot: URL = FileManager.default.temporaryDirectory,
        bundleIdentifier: String = Bundle.main.bundleIdentifier ?? "com.semanticgallery.app"
    ) {
        self.selfRemovalDelaySeconds = selfRemovalDelaySeconds
        self.fileManager = fileManager
        self.bookmarkStore = bookmarkStore
        self.homeRoot = homeRoot
        self.temporaryRoot = temporaryRoot
        self.bundleIdentifier = bundleIdentifier
    }

    public func removeArtifacts(paths: AppPaths, selectedFolder: URL?) throws {
        let ownedURLs = [paths.supportRoot, paths.cachesRoot, paths.logsRoot]

        for url in ownedURLs where url != selectedFolder {
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
        }

        bookmarkStore.clear()
        clearPreferencesDomain()

        for url in residualURLs() where url != selectedFolder {
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
        }

        try removeByHostPreferenceResidue()

        try removeTemporaryResidue(excluding: selectedFolder)
    }

    public func shouldSelfRemove(appBundleURL: URL) -> Bool {
        appBundleURL.pathExtension == "app"
            && appBundleURL.pathComponents.contains("Applications")
            && appBundleURL.lastPathComponent == "SemanticGallery.app"
    }

    public func writeSelfRemovalScript(
        appBundleURL: URL,
        processIdentifier: Int32,
        scriptRoot: URL
    ) throws -> URL {
        try fileManager.createDirectory(at: scriptRoot, withIntermediateDirectories: true)
        let scriptURL = scriptRoot.appending(path: "semanticgallery-uninstall-\(UUID().uuidString).sh")
        let appPath = shellQuotedPath(appBundleURL)
        let scriptPath = shellQuotedPath(scriptURL)
        let cleanupCommands = postExitCleanupCommands()
            .map { "rm -rf \($0)" }
            .joined(separator: "\n")
        let byHostRoot = shellQuotedPath(homeRoot.appending(path: "Library/Preferences/ByHost"))
        let byHostPattern = bundleIdentifier.replacingOccurrences(of: "'", with: "'\"'\"'") + ".*"
        let defaultsDomain = bundleIdentifier.replacingOccurrences(of: "'", with: "'\"'\"'")
        let contents = """
        #!/bin/zsh
        while kill -0 \(processIdentifier) 2>/dev/null; do
          sleep 0.2
        done
        sleep \(selfRemovalDelaySeconds)
        /usr/bin/defaults delete '\(defaultsDomain)' >/dev/null 2>&1 || true
        /usr/bin/defaults -currentHost delete '\(defaultsDomain)' >/dev/null 2>&1 || true
        rm -rf \(appPath)
        \(cleanupCommands)
        if [ -d \(byHostRoot) ]; then
          find \(byHostRoot) -maxdepth 1 -name '\(byHostPattern)' -exec rm -rf {} +
        fi
        /usr/bin/killall cfprefsd >/dev/null 2>&1 || true
        rm -f \(scriptPath)
        """
        try contents.write(to: scriptURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: scriptURL.path(percentEncoded: false))
        return scriptURL
    }

    public func launchSelfRemoval(
        appBundleURL: URL,
        processIdentifier: Int32,
        scriptRoot: URL = FileManager.default.temporaryDirectory
    ) throws {
        let scriptURL = try writeSelfRemovalScript(
            appBundleURL: appBundleURL,
            processIdentifier: processIdentifier,
            scriptRoot: scriptRoot
        )
        let process = Process()
        process.executableURL = URL(filePath: "/bin/zsh")
        process.arguments = [scriptURL.path(percentEncoded: false)]
        try process.run()
    }

    private func residualURLs() -> [URL] {
        [
            homeRoot.appending(path: "Library/Preferences/\(bundleIdentifier).plist"),
            homeRoot.appending(path: "Library/Saved Application State/\(bundleIdentifier).savedState"),
            homeRoot.appending(path: "Library/Containers/\(bundleIdentifier)"),
            homeRoot.appending(path: "Library/Application Scripts/\(bundleIdentifier)"),
        ]
    }

    private func clearPreferencesDomain() {
        let defaults = UserDefaults.standard
        defaults.removePersistentDomain(forName: bundleIdentifier)
        defaults.synchronize()
        CFPreferencesAppSynchronize(bundleIdentifier as CFString)
    }

    private func removeByHostPreferenceResidue() throws {
        let byHostRoot = homeRoot.appending(path: "Library/Preferences/ByHost")
        guard fileManager.fileExists(atPath: byHostRoot.path) else {
            return
        }

        let children = try fileManager.contentsOfDirectory(
            at: byHostRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )

        for child in children where child.lastPathComponent.hasPrefix("\(bundleIdentifier).") {
            try fileManager.removeItem(at: child)
        }
    }

    private func removeTemporaryResidue(excluding selectedFolder: URL?) throws {
        guard fileManager.fileExists(atPath: temporaryRoot.path) else {
            return
        }

        let children = try fileManager.contentsOfDirectory(
            at: temporaryRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for child in children {
            let candidateURLs = [
                child.appending(path: "SemanticGallery"),
                child.appending(path: "Caches/com.semanticgallery.app"),
                child.appending(path: "Logs/SemanticGallery"),
            ]

            for url in candidateURLs where url != selectedFolder {
                if fileManager.fileExists(atPath: url.path) {
                    try fileManager.removeItem(at: url)
                }
            }
        }
    }

    private func postExitCleanupCommands() -> [String] {
        postExitCleanupTargets().map(shellQuotedPath(_:))
    }

    private func postExitCleanupTargets() -> [URL] {
        [
            homeRoot.appending(path: "Library/Application Support/SemanticGallery"),
            homeRoot.appending(path: "Library/Caches/\(bundleIdentifier)"),
            homeRoot.appending(path: "Library/Logs/SemanticGallery"),
        ] + residualURLs()
    }

    private func shellQuotedPath(_ url: URL) -> String {
        let path = url.path(percentEncoded: false).replacingOccurrences(of: "'", with: "'\"'\"'")
        return "'\(path)'"
    }
}
