import Foundation

@MainActor
public final class FolderAccessSession {
    private var activeScopedFolder: URL?

    public init() {}

    public func activate(_ url: URL) {
        if activeScopedFolder == url {
            return
        }

        deactivate()

        if url.startAccessingSecurityScopedResource() {
            activeScopedFolder = url
        }
    }

    public func deactivate() {
        guard let activeScopedFolder else {
            return
        }

        activeScopedFolder.stopAccessingSecurityScopedResource()
        self.activeScopedFolder = nil
    }
}
