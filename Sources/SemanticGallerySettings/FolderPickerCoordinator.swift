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
        return panel.runModal() == .OK ? panel.url : nil
    }
}
