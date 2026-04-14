import Foundation
import SemanticGalleryPersistence

public enum InstallStep: Equatable, Sendable {
    case prepareDirectories
    case prepareDatabase
    case downloadBaseModel
    case downloadStage1Checkpoint
    case downloadPublicAnchor
    case verifyArtifacts
    case finalizeInstallation
}

public struct InstallProgress: Equatable, Sendable {
    public let step: InstallStep
    public let message: String

    public init(step: InstallStep, message: String) {
        self.step = step
        self.message = message
    }
}

public actor InstallCoordinator {
    private let paths: AppPaths
    private let downloader: ArtifactDownloading
    private let catalog: ArtifactCatalog

    public init(
        paths: AppPaths,
        downloader: ArtifactDownloading,
        catalog: ArtifactCatalog = .legacyCompatible
    ) {
        self.paths = paths
        self.downloader = downloader
        self.catalog = catalog
    }

    public func prepare() async throws -> [InstallProgress] {
        var progress: [InstallProgress] = []

        try FileManager.default.createDirectory(at: paths.supportRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: paths.artifactsRoot, withIntermediateDirectories: true)
        progress.append(.init(step: .prepareDirectories, message: "Preparing app directories"))

        if FileManager.default.fileExists(atPath: paths.databaseURL.path) == false {
            FileManager.default.createFile(atPath: paths.databaseURL.path, contents: Data())
        }
        progress.append(.init(step: .prepareDatabase, message: "Preparing local database"))

        try await downloader.download(artifact: catalog.baseModel, into: paths.artifactsRoot)
        progress.append(.init(step: .downloadBaseModel, message: "Downloading SigLIP2 base model"))

        try await downloader.download(artifact: catalog.stage1Checkpoint, into: paths.artifactsRoot)
        progress.append(.init(step: .downloadStage1Checkpoint, message: "Downloading Stage 1 retrieval checkpoint"))

        try await downloader.download(artifact: catalog.publicAnchor, into: paths.artifactsRoot)
        progress.append(.init(step: .downloadPublicAnchor, message: "Downloading public adaptation anchor"))

        try Data(#"{"status":"complete"}"#.utf8).write(to: paths.installStateURL)
        progress.append(.init(step: .verifyArtifacts, message: "Verifying installed artifacts"))

        let installStateStore = InstallStateStore(paths: paths, catalog: catalog)
        guard try installStateStore.isInstallationComplete() else {
            throw CocoaError(.fileReadCorruptFile)
        }

        progress.append(.init(step: .finalizeInstallation, message: "Installation complete"))
        return progress
    }
}
