import AppKit
import Foundation

@MainActor
public struct FolderPickerCoordinator {
    public init() {}

    public func pickFolder() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose Folder"
        panel.directoryURL = Self.initialDirectoryURL()
        return panel.runModal() == .OK ? panel.url : nil
    }

    nonisolated static func initialDirectoryURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL {
        if let uiTestFixtureRoot = environment["SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT"],
           uiTestFixtureRoot.isEmpty == false {
            return URL(filePath: uiTestFixtureRoot, directoryHint: .isDirectory)
        }

        if let picturesDirectory = fileManager.urls(for: .picturesDirectory, in: .userDomainMask).first {
            return picturesDirectory
        }

        return fileManager.homeDirectoryForCurrentUser
    }
}
