import AppKit
import Foundation
import Observation
import SemanticGalleryCore
import SemanticGalleryInstall
import SemanticGalleryML
import SemanticGalleryPersistence
import SemanticGallerySearch
import SemanticGallerySettings

@MainActor
@Observable
public final class GalleryController {
    public let statusState: StatusState
    public let libraryState: LibraryState
    public let installationState: InstallationState
    public let usageState: UsageState
    public let thumbnailCache: ThumbnailCache

    let runtimeOptions: RuntimeConfiguration
    let bookmarkStore: FolderBookmarkStore
    let folderAccessSession: FolderAccessSession
    let folderChangeMonitor: any FolderChangeMonitoring
    let folderPickerCoordinator: FolderPickerCoordinator
    let uninstallCoordinator: UninstallCoordinator
    let installCoordinator: InstallCoordinator
    let baseEmbeddingService: any GalleryEmbeddingService
    let adaptationTrainer: (any GalleryAdaptationTraining)?

    var libraryDatabase: LibraryDatabase?
    var folderSearchIndex: FolderSearchIndex?
    var activeEmbeddingService: any GalleryEmbeddingService
    var activeEncoderVersion: String
    var cachedVectorQuery: CachedVectorQuery?
    var observedFolderSyncTask: Task<Void, Never>?
    var pendingObservedFileChanges = 0
    var folderSearchIndexNeedsRefresh = false
    var lastLoggedFolderPreparationStep: FolderPreparationStep?
    var lastLoggedFolderPreparationAt = Date.distantPast

    public init(
        runtimeOptions: RuntimeConfiguration = .current(),
        embeddingService: (any GalleryEmbeddingService)? = nil,
        adaptationTrainer: (any GalleryAdaptationTraining)? = nil,
        folderChangeMonitor: (any FolderChangeMonitoring)? = nil
    ) {
        let bookmarkStore = FolderBookmarkStore(
            bookmarkFileURL: runtimeOptions.paths.supportRoot.appending(path: "selected-folder.bookmark")
        )

        self.runtimeOptions = runtimeOptions
        self.statusState = StatusState()
        self.libraryState = LibraryState()
        self.installationState = InstallationState()
        self.usageState = UsageState()
        self.thumbnailCache = ThumbnailCache(cacheRoot: runtimeOptions.paths.cachesRoot.appending(path: "Thumbnails"))
        self.bookmarkStore = bookmarkStore
        self.folderAccessSession = FolderAccessSession()
        self.folderChangeMonitor = folderChangeMonitor ?? FolderChangeMonitor()
        self.folderPickerCoordinator = FolderPickerCoordinator()
        self.uninstallCoordinator = UninstallCoordinator(
            bookmarkStore: bookmarkStore
        )

        let installDownloader: ArtifactDownloading
        if runtimeOptions.useStubDownloads {
            installDownloader = ArtifactDownloader.stubbed
        } else if let artifactSourceRoot = runtimeOptions.artifactSourceRoot {
            installDownloader = ArtifactDownloader(
                mode: .live,
                fileSources: ArtifactFileSourceCatalog(root: artifactSourceRoot).fileSources
            )
        } else {
            installDownloader = ArtifactDownloader()
        }

        self.installCoordinator = InstallCoordinator(
            paths: runtimeOptions.paths,
            downloader: installDownloader
        )

        let resolvedEmbeddingService = embeddingService ?? Stage1SigLIP2EmbeddingService(paths: runtimeOptions.paths)
        self.baseEmbeddingService = resolvedEmbeddingService
        self.activeEmbeddingService = resolvedEmbeddingService
        self.activeEncoderVersion = resolvedEmbeddingService.encoderVersion
        self.adaptationTrainer = adaptationTrainer ?? PrivateAdaptationTrainer(paths: runtimeOptions.paths)

        ensureRuntimeDirectoriesExist()
        bootstrap()
    }
}
