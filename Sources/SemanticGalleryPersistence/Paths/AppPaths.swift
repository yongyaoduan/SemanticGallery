import Foundation

public struct AppPaths: Sendable {
    public let root: URL
    public let supportRoot: URL
    public let cachesRoot: URL
    public let logsRoot: URL
    public let databaseURL: URL
    public let installStateURL: URL
    public let artifactsRoot: URL
    public let modelsRoot: URL
    public let datasetsRoot: URL
    public let bundledArtifactsRoot: URL?

    public init(root: URL? = nil, bundledArtifactsRoot: URL? = AppPaths.defaultBundledArtifactsRoot()) {
        if let root {
            self.root = root.appending(path: "SemanticGallery")
            self.supportRoot = self.root
            self.cachesRoot = self.root.appending(path: "Caches")
            self.logsRoot = self.root.appending(path: "Logs")
        } else {
            let fileManager = FileManager.default
            let supportBase = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]

            self.root = supportBase.appending(path: "SemanticGallery")
            self.supportRoot = self.root
            self.cachesRoot = self.root.appending(path: "Caches")
            self.logsRoot = self.root.appending(path: "Logs")
        }
        self.databaseURL = self.root.appending(path: "library.sqlite")
        self.installStateURL = self.root.appending(path: "install-state.json")
        self.artifactsRoot = self.root.appending(path: "Artifacts")
        self.modelsRoot = self.root.appending(path: "Models")
        self.datasetsRoot = self.root.appending(path: "Datasets")
        self.bundledArtifactsRoot = bundledArtifactsRoot
    }

    public init(
        supportRoot: URL,
        cachesRoot: URL,
        logsRoot: URL,
        bundledArtifactsRoot: URL? = AppPaths.defaultBundledArtifactsRoot()
    ) {
        self.root = supportRoot
        self.supportRoot = supportRoot
        self.cachesRoot = cachesRoot
        self.logsRoot = logsRoot
        self.databaseURL = supportRoot.appending(path: "library.sqlite")
        self.installStateURL = supportRoot.appending(path: "install-state.json")
        self.artifactsRoot = supportRoot.appending(path: "Artifacts")
        self.modelsRoot = supportRoot.appending(path: "Models")
        self.datasetsRoot = supportRoot.appending(path: "Datasets")
        self.bundledArtifactsRoot = bundledArtifactsRoot
    }

    public var runtimeArtifactsRoot: URL {
        bundledArtifactsRoot ?? artifactsRoot
    }

    public static func defaultBundledArtifactsRoot(
        bundle: Bundle = .main,
        fileManager: FileManager = .default
    ) -> URL? {
        let candidates = [
            bundle.resourceURL?.appending(path: "SemanticGalleryArtifacts"),
            bundle.bundleURL.appending(path: "Contents").appending(path: "Resources").appending(path: "SemanticGalleryArtifacts"),
        ]
        return candidates
            .compactMap { $0 }
            .first(where: { fileManager.fileExists(atPath: $0.path) })
    }
}
