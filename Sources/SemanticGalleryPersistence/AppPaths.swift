import Foundation

public struct AppPaths: Sendable {
    public let root: URL
    public let supportRoot: URL
    public let cachesRoot: URL
    public let logsRoot: URL
    public let databaseURL: URL
    public let installStateURL: URL
    public let artifactsRoot: URL

    public init(root: URL? = nil) {
        let fileManager = FileManager.default
        if let root {
            self.root = root.appending(path: "SemanticGallery")
            self.supportRoot = self.root
            self.cachesRoot = root.appending(path: "Caches").appending(path: "com.semanticgallery.app")
            self.logsRoot = root.appending(path: "Logs").appending(path: "SemanticGallery")
        } else {
            let supportBase = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            let cachesBase = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            let logsBase = fileManager.urls(for: .libraryDirectory, in: .userDomainMask)[0].appending(path: "Logs")

            self.root = supportBase.appending(path: "SemanticGallery")
            self.supportRoot = self.root
            self.cachesRoot = cachesBase.appending(path: "com.semanticgallery.app")
            self.logsRoot = logsBase.appending(path: "SemanticGallery")
        }
        self.databaseURL = self.root.appending(path: "library.sqlite")
        self.installStateURL = self.root.appending(path: "install-state.json")
        self.artifactsRoot = self.root.appending(path: "Artifacts")
    }
}
