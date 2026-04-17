import AppKit
import Foundation
import Observation
import SemanticGalleryCore
import SemanticGalleryIndex
import SemanticGalleryInstall
import SemanticGalleryML
import SemanticGalleryPersistence
import SemanticGallerySearch
import SemanticGallerySettings

@MainActor
@Observable
public final class SemanticGalleryController {
    public let statusStore: AppStatusStore
    public let libraryStateStore: LibraryStateStore
    public let installationStateStore: InstallationStateStore
    public let workspaceStateStore: WorkspaceStateStore
    public let thumbnailStore: ThumbnailStore

    private let runtimeOptions: SemanticGalleryRuntimeOptions
    private let bookmarkStore: FolderBookmarkStore
    private let folderAccessSession: FolderAccessSession
    private let folderChangeMonitor: any FolderChangeMonitoring
    private let folderPickerCoordinator: FolderPickerCoordinator
    private let uninstallCoordinator: UninstallCoordinator
    private let installCoordinator: InstallCoordinator
    private let installStateStore: InstallStateStore
    private let baseEmbeddingService: any GalleryEmbeddingService
    private let adaptationTrainer: (any GalleryAdaptationTraining)?

    private var libraryDatabase: LibraryDatabase?
    private var activeSearchView: ActiveSearchView?
    private var activeEmbeddingService: any GalleryEmbeddingService
    private var activeEncoderVersion: String
    private var cachedVectorQuery: CachedVectorQuery?
    private var observedFolderSyncTask: Task<Void, Never>?
    private var pendingObservedFileChanges = 0
    private var activeViewNeedsRefresh = false
    private var lastLoggedFolderPreparationStep: FolderPreparationStep?
    private var lastLoggedFolderPreparationAt = Date.distantPast

    public init(
        runtimeOptions: SemanticGalleryRuntimeOptions = .current(),
        embeddingService: (any GalleryEmbeddingService)? = nil,
        adaptationTrainer: (any GalleryAdaptationTraining)? = nil,
        folderChangeMonitor: (any FolderChangeMonitoring)? = nil
    ) {
        let bookmarkStore = FolderBookmarkStore(
            bookmarkFileURL: runtimeOptions.paths.supportRoot.appending(path: "selected-folder.bookmark")
        )

        self.runtimeOptions = runtimeOptions
        self.statusStore = AppStatusStore()
        self.libraryStateStore = LibraryStateStore()
        self.installationStateStore = InstallationStateStore()
        self.workspaceStateStore = WorkspaceStateStore()
        self.thumbnailStore = ThumbnailStore(cacheRoot: runtimeOptions.paths.cachesRoot.appending(path: "Thumbnails"))
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
        self.installStateStore = InstallStateStore(paths: runtimeOptions.paths)
        let resolvedEmbeddingService = embeddingService ?? Stage1SigLIP2EmbeddingService(paths: runtimeOptions.paths)
        self.baseEmbeddingService = resolvedEmbeddingService
        self.activeEmbeddingService = resolvedEmbeddingService
        self.activeEncoderVersion = resolvedEmbeddingService.encoderVersion
        self.adaptationTrainer = adaptationTrainer ?? PrivateAdaptationTrainer(paths: runtimeOptions.paths)

        ensureRuntimeDirectoriesExist()
        bootstrap()
    }

    public func startInstallation() {
        guard statusStore.status != .installing else {
            return
        }

        let installStepDelayNanoseconds = runtimeOptions.installStepDelayNanoseconds
        statusStore.status = .installing
        installationStateStore.progress = InstallProgress.placeholders()

        Task {
            do {
                let progress = try await installCoordinator.prepare { item in
                    await MainActor.run {
                        self.replaceInstallProgressLocally(item)
                    }
                    if installStepDelayNanoseconds > 0 {
                        try? await Task.sleep(nanoseconds: installStepDelayNanoseconds)
                    }
                }

                await MainActor.run {
                    self.installationStateStore.progress = progress
                    self.statusStore.status = .installComplete
                }
                await self.prewarmActiveEmbeddingService()
            } catch {
                await MainActor.run {
                    self.installationStateStore.progress = []
                    self.statusStore.status = .installFailed("Installation could not finish.")
                }
            }
        }
    }

    public func enterWorkspace() {
        statusStore.status = libraryStateStore.selectedFolder == nil ? .readyWithoutFolder : .ready
    }

    public func chooseFolder() {
        guard let url = folderPickerCoordinator.pickFolder() else {
            return
        }

        Task {
            await selectFolder(at: url)
        }
    }

    public func selectFolder(at url: URL) async {
        let previousSelectedFolder = libraryStateStore.selectedFolder
        startFolderObservation(for: nil)
        statusStore.status = .indexing
        ensureRuntimeDirectoriesExist()
        await configureEmbeddingService(for: url)
        beginFolderSelection(at: url)

        do {
            let database = try ensureDatabase()
            let outcome = try await prepareFolderSelection(
                at: url,
                database: database,
                embeddingService: activeEmbeddingService,
                encoderVersion: activeEncoderVersion,
                stepDelayNanoseconds: runtimeOptions.folderPreparationDelayNanoseconds
            ) { progress in
                await MainActor.run {
                    if self.libraryStateStore.folderPreparationProgress.contains(where: { $0.step == progress.step }) {
                        self.replaceFolderProgressLocally(progress)
                    } else {
                        self.libraryStateStore.folderPreparationProgress.append(progress)
                    }
                    self.recordFolderPreparationProgress(progress, for: url)
                }
            }
            activeSearchView = try? searchCoordinator(database: database).activeSearchView(folderAbsolutePath: outcome.folderPath)
            try await refreshVisibleResults(for: url)
            applyLibrarySearchState(outcome.searchState)
            libraryStateStore.activeEncoderVersion = activeEncoderVersion
            try bookmarkStore.saveBookmark(for: url)
            finishPreparedFolderSelection(at: url)
            recordRuntimeLog(
                "Folder preparation finished.",
                category: "library",
                metadata: folderPreparationMetadata(
                    for: url,
                    outcome: outcome,
                    encoderVersion: activeEncoderVersion
                )
            )
            if outcome.skippedImageCount > 0 {
                recordRuntimeLog(
                    "Some images were skipped because they could not be decoded for semantic search.",
                    level: .warning,
                    category: "library",
                    metadata: folderPreparationMetadata(
                        for: url,
                        outcome: outcome,
                        encoderVersion: activeEncoderVersion
                    )
                )
            }
            await prewarmActiveEmbeddingService()
        } catch let error as GalleryEmbeddingError {
            libraryStateStore.folderPreparationProgress = []
            recordFolderSelectionFailure(error, at: url)
            switch error {
            case .invalidModelArtifact, .invalidStage1Weights:
                await finishFailedFolderSelection(
                    at: url,
                    previousSelectedFolder: previousSelectedFolder,
                    searchIssue: searchIssueMessage(for: error)
                )
            case .invalidAdapterArtifact:
                activeEmbeddingService = baseEmbeddingService
                activeEncoderVersion = baseEmbeddingService.encoderVersion
                libraryStateStore.activeEncoderVersion = activeEncoderVersion
                cachedVectorQuery = nil
                libraryStateStore.privateAdaptationNotice = "The local adaptation could not be loaded, so SemanticGallery returned to the published encoder."
                await finishFailedFolderSelection(at: url, previousSelectedFolder: previousSelectedFolder)
            case .invalidImageData:
                await finishFailedFolderSelection(
                    at: url,
                    previousSelectedFolder: previousSelectedFolder,
                    searchIssue: searchIssueMessage(for: error)
                )
            }
        } catch {
            libraryStateStore.folderPreparationProgress = []
            recordFolderSelectionFailure(error, at: url, includeDetails: true)
            await finishFailedFolderSelection(
                at: url,
                previousSelectedFolder: previousSelectedFolder,
                searchIssue: searchIssueMessage(for: error)
            )
        }
    }

    public func runSearch() {
        guard let selectedFolder = libraryStateStore.selectedFolder else {
            return
        }

        Task {
            do {
                try await refreshVisibleResults(for: selectedFolder)
            } catch {
                await MainActor.run {
                    self.workspaceStateStore.results = []
                    self.workspaceStateStore.clearSelection()
                }
            }
        }
    }

    public func startPrivateAdaptation() {
        guard
            let selectedFolder = libraryStateStore.selectedFolder,
            let adaptationTrainer
        else {
            return
        }

        let supportedImageCount = libraryStateStore.supportedImageCount ?? 0
        guard supportedImageCount >= 100 else {
            libraryStateStore.privateAdaptationNotice =
                "This folder currently has \(supportedImageCount) supported images. SemanticGallery needs at least 100 images before local adaptation can begin."
            libraryStateStore.privateAdaptationProgress = []
            statusStore.status = .ready
            return
        }

        libraryStateStore.privateAdaptationNotice = nil
        libraryStateStore.privateAdaptationProgress = PrivateAdaptationProgress.placeholders()
        statusStore.status = .training

        let trainingRunID: Int64?
        do {
            let database = try ensureDatabase()
            let folderID = try database.upsertFolder(
                absolutePath: selectedFolder.path(percentEncoded: false),
                bookmarkData: nil,
                isActive: true
            )
            trainingRunID = try database.insertTrainingRun(
                folderID: folderID,
                encoderVersion: activeEncoderVersion,
                status: "running",
                summaryJSON: nil
            )
        } catch {
            trainingRunID = nil
        }

        Task {
            do {
                let adaptation = try await adaptationTrainer.runAdaptation(for: selectedFolder) { progress in
                    await MainActor.run {
                        self.replacePrivateAdaptationProgress(progress)
                    }
                }

                await MainActor.run {
                    self.activeEmbeddingService = adaptation.embeddingService
                    self.activeEncoderVersion = adaptation.artifact.encoderVersion
                    self.libraryStateStore.activeEncoderVersion = adaptation.artifact.encoderVersion
                    self.cachedVectorQuery = nil
                }

                try await self.rebuildActiveEncoderIndex(for: selectedFolder)

                if let trainingRunID {
                    let database = try self.ensureDatabase()
                    let summaryJSON = try? String(contentsOf: adaptation.artifact.summaryURL, encoding: .utf8)
                    try database.finishTrainingRun(
                        id: trainingRunID,
                        status: "completed",
                        summaryJSON: summaryJSON
                    )
                }

                await MainActor.run {
                    self.statusStore.status = .ready
                }
            } catch {
                if let trainingRunID, let database = try? self.ensureDatabase() {
                    try? database.finishTrainingRun(
                        id: trainingRunID,
                        status: "failed",
                        summaryJSON: nil
                    )
                }
                await MainActor.run {
                    self.libraryStateStore.privateAdaptationProgress = []
                    self.libraryStateStore.privateAdaptationNotice = "Private adaptation could not finish."
                    self.cachedVectorQuery = nil
                    self.statusStore.status = .ready
                }
            }
        }
    }

    public func toggleSelection(assetID: Int64) {
        workspaceStateStore.toggleSelection(assetID: assetID)
    }

    public func enterSelectionMode() {
        workspaceStateStore.enterSelectionMode()
    }

    public func leaveSelectionMode() {
        workspaceStateStore.leaveSelectionMode()
    }

    public func selectAllVisibleResults() {
        workspaceStateStore.selectAllVisible()
    }

    public func clearSelection() {
        workspaceStateStore.clearSelection()
    }

    public func openPreview(assetID: Int64) {
        workspaceStateStore.openPreview(assetID: assetID)
    }

    public func closePreview() {
        workspaceStateStore.closePreview()
    }

    public func togglePreviewMetadata() {
        workspaceStateStore.togglePreviewMetadata()
    }

    public func searchSimilarToPreviewItem() {
        guard
            let previewItem = workspaceStateStore.previewItem,
            let selectedFolder = libraryStateStore.selectedFolder,
            let imageData = try? Data(contentsOf: URL(filePath: previewItem.absolutePath))
        else {
            return
        }

        workspaceStateStore.queryText = ""
        workspaceStateStore.pastedImageData = imageData
        workspaceStateStore.closePreview()

        Task {
            do {
                try await refreshVisibleResults(for: selectedFolder)
            } catch {
                await MainActor.run {
                    self.workspaceStateStore.results = []
                    self.workspaceStateStore.clearSelection()
                }
            }
        }
    }

    public func showNextPreviewItem() {
        workspaceStateStore.showNextPreviewItem()
    }

    public func showPreviousPreviewItem() {
        workspaceStateStore.showPreviousPreviewItem()
    }

    public func deletePreviewItem() {
        guard let previewAssetID = workspaceStateStore.previewAssetID else {
            return
        }

        Task {
            await performDeleteItems(withIDs: [previewAssetID])
        }
    }

    public func deleteSelection() {
        Task {
            await performDeleteItems(withIDs: workspaceStateStore.selectedAssetIDs)
        }
    }

    public func startUninstall() {
        statusStore.status = .uninstalling
        let selectedFolder = libraryStateStore.selectedFolder
        startFolderObservation(for: nil)

        libraryStateStore.selectedFolder = nil
        libraryStateStore.folderPreparationProgress = []
        installationStateStore.progress = []
        workspaceStateStore.results = []
        workspaceStateStore.pastedImageData = nil
        workspaceStateStore.closePreview()
        workspaceStateStore.leaveSelectionMode()
        workspaceStateStore.clearSelection()
        libraryStateStore.privateAdaptationProgress = []
        libraryStateStore.privateAdaptationNotice = nil
        libraryStateStore.supportedImageCount = nil
        libraryStateStore.searchableImageCount = nil
        libraryStateStore.activeEncoderVersion = nil
        libraryStateStore.searchIssue = nil
        activeSearchView = nil
        libraryDatabase = nil
        activeEmbeddingService = baseEmbeddingService
        activeEncoderVersion = baseEmbeddingService.encoderVersion
        cachedVectorQuery = nil
        folderAccessSession.deactivate()

        do {
            try uninstallCoordinator.removeArtifacts(
                paths: runtimeOptions.paths,
                selectedFolder: selectedFolder
            )
            let appBundleURL = Bundle.main.bundleURL
            if uninstallCoordinator.shouldSelfRemove(appBundleURL: appBundleURL) {
                try uninstallCoordinator.launchSelfRemoval(
                    appBundleURL: appBundleURL,
                    processIdentifier: ProcessInfo.processInfo.processIdentifier
                )
                if runtimeOptions.deferSelfUninstallTermination {
                    return
                }
                NSApplication.shared.terminate(nil)
            } else {
                statusStore.status = .readyWithoutFolder
            }
        } catch {
            statusStore.status = .installFailed("Unable to remove SemanticGallery data.")
        }
    }

    private func bootstrap() {
        do {
            if let selectedFolder = try bookmarkStore.loadBookmark() {
                folderAccessSession.activate(selectedFolder)
                libraryStateStore.selectedFolder = selectedFolder
                libraryStateStore.activeEncoderVersion = activeEncoderVersion
                Task {
                    await self.configureEmbeddingService(for: selectedFolder)
                    let shouldReindex = self.shouldReindexRememberedFolder(selectedFolder)
                    await MainActor.run {
                        self.statusStore.status = shouldReindex ? .indexing : .ready
                    }
                    do {
                        try await self.rebuildWorkspace(for: selectedFolder, shouldReindex: shouldReindex)
                        await MainActor.run {
                            self.statusStore.status = .ready
                        }
                        self.startFolderObservation(for: selectedFolder)
                    } catch {
                        await MainActor.run {
                            self.folderAccessSession.deactivate()
                            self.libraryStateStore.selectedFolder = nil
                            self.libraryStateStore.supportedImageCount = nil
                            self.libraryStateStore.searchableImageCount = nil
                            self.workspaceStateStore.results = []
                            self.workspaceStateStore.clearSelection()
                            self.workspaceStateStore.leaveSelectionMode()
                            self.workspaceStateStore.closePreview()
                            self.workspaceStateStore.pastedImageData = nil
                            self.libraryStateStore.folderPreparationProgress = []
                            self.libraryStateStore.searchIssue = nil
                            self.statusStore.status = .readyWithoutFolder
                        }
                        self.bookmarkStore.clear()
                        self.startFolderObservation(for: nil)
                    }
                    await self.prewarmActiveEmbeddingService()
                }
            } else {
                startFolderObservation(for: nil)
                statusStore.status = .readyWithoutFolder
                Task {
                    await self.prewarmActiveEmbeddingService()
                }
            }
        } catch {
            startFolderObservation(for: nil)
            statusStore.status = .readyWithoutFolder
        }
    }

    private func rebuildWorkspace(for url: URL, shouldReindex: Bool = true) async throws {
        let database = try ensureDatabase()
        if shouldReindex {
            let summary = try await FolderIndexer(database: database).rebuildIndex(for: url)
            try await populateMissingEmbeddings(
                inFolderAbsolutePath: summary.folderPath,
                database: database
            )
            applyLibrarySearchState(try folderSearchState(for: summary.folderPath, database: database))
        } else {
            applyLibrarySearchState(try folderSearchState(for: url.path(percentEncoded: false), database: database))
        }

        try refreshActiveSearchView(for: url)
        workspaceStateStore.pastedImageData = nil
        try await refreshVisibleResults(for: url)
    }

    private func refreshVisibleResults(for url: URL) async throws {
        let database = try ensureDatabase()
        let coordinator = SearchCoordinator(
            database: database,
            encoderVersion: activeEncoderVersion
        )
        workspaceStateStore.isSearching = true
        let previousStatus = statusStore.status
        if previousStatus != .indexing && previousStatus != .training {
            statusStore.status = .searching
        }
        defer {
            workspaceStateStore.isSearching = false
            if self.statusStore.status == .searching {
                self.statusStore.status = previousStatus == .searching
                    ? (self.libraryStateStore.selectedFolder == nil ? .readyWithoutFolder : .ready)
                    : previousStatus
            }
        }

        let folderPath = url.path(percentEncoded: false)
        let trimmedQuery = workspaceStateStore.queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let queryLimit = workspaceStateStore.resultLimit

        if let pastedImageData = workspaceStateStore.pastedImageData {
            let queryVector = try await vectorQuery(forImageData: pastedImageData)
            try ensureActiveSearchViewIsCurrent(for: url)

            if let activeSearchView, activeSearchView.items.isEmpty == false {
                workspaceStateStore.results = coordinator.vectorSearch(
                    activeView: activeSearchView,
                    queryVector: queryVector,
                    limit: queryLimit
                )
            } else {
                recordSemanticSearchUnavailable(
                    for: url,
                    queryType: "image"
                )
                workspaceStateStore.results = []
            }
        } else if trimmedQuery.isEmpty == false {
            let lexicalMatches = try coordinator.search(
                folderAbsolutePath: folderPath,
                query: trimmedQuery,
                limit: queryLimit
            )

            if lexicalMatches.isEmpty == false {
                workspaceStateStore.results = lexicalMatches
            } else {
                let queryVector = try await vectorQuery(forText: trimmedQuery)
                try ensureActiveSearchViewIsCurrent(for: url)

                if let activeSearchView, activeSearchView.items.isEmpty == false {
                    workspaceStateStore.results = coordinator.vectorSearch(
                        activeView: activeSearchView,
                        queryVector: queryVector,
                        limit: queryLimit
                    )
                } else {
                    recordSemanticSearchUnavailable(
                        for: url,
                        queryType: "text"
                    )
                    workspaceStateStore.results = []
                }
            }
        } else {
            workspaceStateStore.results = try coordinator.search(
                folderAbsolutePath: folderPath,
                query: trimmedQuery,
                limit: queryLimit
            )
        }

        if workspaceStateStore.previewItem == nil {
            workspaceStateStore.closePreview()
        }
        workspaceStateStore.clearSelection()
    }

    private func populateMissingEmbeddings(
        inFolderAbsolutePath folderAbsolutePath: String,
        database: LibraryDatabase
    ) async throws {
        let missingEmbeddings = try database.missingEmbeddingWork(
            inFolderAbsolutePath: folderAbsolutePath,
            encoderVersion: activeEncoderVersion
        )

        for item in missingEmbeddings {
            do {
                let vector = try await activeEmbeddingService.encodeImage(at: URL(filePath: item.representativeAbsolutePath))
                try database.upsertEmbedding(
                    assetID: item.assetID,
                    encoderVersion: activeEncoderVersion,
                    vector: vector
                )
            } catch let error as GalleryEmbeddingError where error == .invalidImageData {
                continue
            }
        }
    }

    private func ensureActiveSearchViewIsCurrent(for url: URL) throws {
        if activeSearchView == nil || activeViewNeedsRefresh {
            try refreshActiveSearchView(for: url)
        }
    }

    private func refreshActiveSearchView(for url: URL) throws {
        let coordinator = searchCoordinator(database: try ensureDatabase())
        activeSearchView = try coordinator.activeSearchView(folderAbsolutePath: url.path(percentEncoded: false))
        pendingObservedFileChanges = 0
        activeViewNeedsRefresh = false
    }

    private func folderSearchState(
        for folderPath: String,
        database: LibraryDatabase
    ) throws -> FolderSearchState {
        FolderSearchState(
            supportedImageCount: try database.visibleFileInstances(
                inFolderAbsolutePath: folderPath,
                limit: Int.max
            ).count,
            searchableImageCount: try database.embeddedFileCount(
                inFolderAbsolutePath: folderPath,
                encoderVersion: activeEncoderVersion
            )
        )
    }

    private func applyLibrarySearchState(
        _ state: FolderSearchState,
        searchIssue: String? = nil
    ) {
        libraryStateStore.supportedImageCount = state.supportedImageCount
        libraryStateStore.searchableImageCount = state.searchableImageCount

        if let searchIssue, state.searchableImageCount < state.supportedImageCount {
            libraryStateStore.searchIssue = searchIssue
        } else if state.supportedImageCount > 0, state.searchableImageCount == 0 {
            libraryStateStore.searchIssue = "Semantic search could not prepare embeddings for this folder."
        } else {
            libraryStateStore.searchIssue = nil
        }
    }

    private func recordRuntimeLog(
        _ message: String,
        level: SemanticGalleryRuntimeLogLevel = .info,
        category: String,
        metadata: [String: String] = [:]
    ) {
        SemanticGalleryRuntimeLog.record(
            message,
            level: level,
            category: category,
            paths: runtimeOptions.paths,
            metadata: metadata
        )
    }

    private func resetFolderPreparationRuntimeLogState() {
        lastLoggedFolderPreparationStep = nil
        lastLoggedFolderPreparationAt = .distantPast
    }

    private func recordFolderPreparationProgress(_ progress: FolderPreparationProgress, for url: URL) {
        let timestamp = progress.recordedAt ?? Date()
        let stepChanged = lastLoggedFolderPreparationStep != progress.step
        let enoughTimeElapsed = timestamp.timeIntervalSince(lastLoggedFolderPreparationAt) >= 10
        let completedStep = progress.stepProgress >= 1.0
        guard stepChanged || enoughTimeElapsed || completedStep else {
            return
        }

        lastLoggedFolderPreparationStep = progress.step
        lastLoggedFolderPreparationAt = timestamp
        recordRuntimeLog(
            progress.message,
            category: "library",
            metadata: [
                "elapsed_seconds": progress.elapsedSeconds.map(String.init) ?? "",
                "encoder_version": activeEncoderVersion,
                "folder": url.path(percentEncoded: false),
                "overall_progress_percent": "\(Int((progress.overallProgress * 100).rounded()))",
                "remaining_seconds": progress.remainingSeconds.map(String.init) ?? "",
                "step": folderPreparationStepIdentifier(progress.step),
                "step_progress_percent": "\(Int((progress.stepProgress * 100).rounded()))",
            ]
        )
    }

    private func folderPreparationStepIdentifier(_ step: FolderPreparationStep) -> String {
        switch step {
        case .requestFolderAccess:
            return "request_folder_access"
        case .scanSupportedImages:
            return "scan_supported_images"
        case .reuseExistingEmbeddings:
            return "reuse_existing_embeddings"
        case .generateEmbeddings:
            return "generate_embeddings"
        case .buildSearchView:
            return "build_search_view"
        case .finalizeFolderSelection:
            return "finalize_folder_selection"
        }
    }

    private func recordSemanticSearchUnavailable(for url: URL, queryType: String) {
        recordRuntimeLog(
            "Semantic search could not run because no searchable embeddings are available.",
            level: .warning,
            category: "search",
            metadata: [
                "encoder_version": activeEncoderVersion,
                "folder": url.path(percentEncoded: false),
                "query_type": queryType,
                "searchable_images": "\(libraryStateStore.searchableImageCount ?? 0)",
                "supported_images": "\(libraryStateStore.supportedImageCount ?? 0)",
            ]
        )
    }

    private func ensureDatabase() throws -> LibraryDatabase {
        if let libraryDatabase {
            try libraryDatabase.migrate()
            return libraryDatabase
        }

        ensureRuntimeDirectoriesExist()
        let fileManager = FileManager.default
        let databaseURL = runtimeOptions.paths.databaseURL
        if fileManager.fileExists(atPath: databaseURL.path(percentEncoded: false)),
           let fileSize = try? databaseURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           fileSize == 0 {
            recordRuntimeLog(
                "An empty library database was removed before reopening it.",
                level: .warning,
                category: "database",
                metadata: [
                    "database": databaseURL.path(percentEncoded: false),
                ]
            )
            try? fileManager.removeItem(at: databaseURL)
        }

        let database = try LibraryDatabase.open(at: runtimeOptions.paths.databaseURL)
        try database.migrate()
        libraryDatabase = database
        return database
    }

    private func searchCoordinator(database: LibraryDatabase) -> SearchCoordinator {
        SearchCoordinator(
            database: database,
            encoderVersion: activeEncoderVersion
        )
    }

    private func ensureRuntimeDirectoriesExist() {
        RuntimeDirectoryBootstrapper.ensureExists(for: runtimeOptions.paths)
    }

    private func shouldReindexRememberedFolder(_ url: URL) -> Bool {
        let fileManager = FileManager.default
        let databaseURL = runtimeOptions.paths.databaseURL
        guard fileManager.fileExists(atPath: databaseURL.path(percentEncoded: false)) else {
            return true
        }

        if let resourceValues = try? databaseURL.resourceValues(forKeys: [.fileSizeKey]),
           let fileSize = resourceValues.fileSize,
           fileSize == 0 {
            return true
        }

        do {
            let database = try ensureDatabase()
            return try database.containsFolder(absolutePath: url.path(percentEncoded: false)) == false
        } catch {
            return true
        }
    }

    private func startFolderObservation(for url: URL?) {
        observedFolderSyncTask?.cancel()
        observedFolderSyncTask = nil
        pendingObservedFileChanges = 0
        activeViewNeedsRefresh = false
        folderChangeMonitor.stopMonitoring()

        guard let url else {
            return
        }

        try? folderChangeMonitor.startMonitoring(folderURL: url) { [weak self] changeCount in
            Task { @MainActor in
                await self?.recordObservedFolderChanges(changeCount)
            }
        }
    }

    private func recoverPreparedFolderSelection(
        at url: URL,
        searchIssue: String? = nil
    ) async -> Bool {
        let folderPath = url.path(percentEncoded: false)

        guard let database = try? ensureDatabase(),
              (try? database.containsFolder(absolutePath: folderPath)) == true else {
            return false
        }

        let visibleCount = (try? database.visibleFileInstances(
            inFolderAbsolutePath: folderPath,
            limit: Int.max
        ).count) ?? 0
        guard visibleCount > 0 else {
            return false
        }

        folderAccessSession.activate(url)
        libraryStateStore.selectedFolder = url
        let coordinator = searchCoordinator(database: database)
        applyLibrarySearchState(
            (try? folderSearchState(for: folderPath, database: database))
                ?? FolderSearchState(supportedImageCount: visibleCount, searchableImageCount: 0),
            searchIssue: searchIssue
        )
        libraryStateStore.activeEncoderVersion = activeEncoderVersion
        cachedVectorQuery = nil
        activeSearchView = try? coordinator.activeSearchView(folderAbsolutePath: folderPath)

        do {
            try await refreshVisibleResults(for: url)
        } catch {
            let lexicalQuery = workspaceStateStore.queryText.trimmingCharacters(in: .whitespacesAndNewlines)
            workspaceStateStore.results = (try? coordinator.search(
                folderAbsolutePath: folderPath,
                query: lexicalQuery,
                limit: workspaceStateStore.resultLimit
            )) ?? []
            workspaceStateStore.clearSelection()
            if workspaceStateStore.previewItem == nil {
                workspaceStateStore.closePreview()
            }
        }

        try? bookmarkStore.saveBookmark(for: url)
        finishPreparedFolderSelection(at: url)
        recordRuntimeLog(
            searchIssue == nil
                ? "Recovered the selected folder from the existing index."
                : "Recovered visible images from the existing index while semantic search remained unavailable.",
            level: searchIssue == nil ? .info : .warning,
            category: "library",
            metadata: [
                "encoder_version": activeEncoderVersion,
                "folder": folderPath,
                "searchable_images": "\(libraryStateStore.searchableImageCount ?? 0)",
                "supported_images": "\(libraryStateStore.supportedImageCount ?? visibleCount)",
            ]
        )
        return true
    }

    private func recordObservedFolderChanges(_ changeCount: Int) async {
        guard let selectedFolder = libraryStateStore.selectedFolder else {
            return
        }

        pendingObservedFileChanges += max(changeCount, 1)
        activeViewNeedsRefresh = true

        guard observedFolderSyncTask == nil else {
            return
        }

        observedFolderSyncTask = Task { [weak self, selectedFolder] in
            let delay = self?.runtimeOptions.folderObservationDebounceNanoseconds ?? 0
            if delay > 0 {
                try? await Task.sleep(nanoseconds: delay)
            }
            await self?.performObservedFolderSync(for: selectedFolder)
        }
    }

    private func performObservedFolderSync(for url: URL) async {
        defer { observedFolderSyncTask = nil }

        guard let selectedFolder = libraryStateStore.selectedFolder,
              selectedFolder.standardizedFileURL == url.standardizedFileURL else {
            return
        }

        statusStore.status = .indexing
        libraryStateStore.folderPreparationProgress = FolderPreparationProgress.placeholders()
        resetFolderPreparationRuntimeLogState()
        recordRuntimeLog(
            "Observed changes in the selected folder. Refreshing the search index.",
            category: "library",
            metadata: [
                "encoder_version": activeEncoderVersion,
                "folder": url.path(percentEncoded: false),
                "pending_change_count": "\(pendingObservedFileChanges)",
            ]
        )

        do {
            let database = try ensureDatabase()
            let outcome = try await prepareFolderSelection(
                at: url,
                database: database,
                embeddingService: activeEmbeddingService,
                encoderVersion: activeEncoderVersion,
                stepDelayNanoseconds: 0
            ) { progress in
                await MainActor.run {
                    self.replaceFolderProgressLocally(progress)
                    self.recordFolderPreparationProgress(progress, for: url)
                }
            }

            applyLibrarySearchState(outcome.searchState)
            if FolderSyncPolicy.shouldRebuildActiveView(
                currentViewExists: activeSearchView != nil,
                pendingChangeCount: pendingObservedFileChanges,
                visibleImageCount: outcome.supportedImageCount
            ) {
                try refreshActiveSearchView(for: url)
            }
            try await refreshVisibleResults(for: url)
            libraryStateStore.folderPreparationProgress = []
            statusStore.status = .ready
            recordRuntimeLog(
                "Observed folder changes were indexed successfully.",
                category: "library",
                metadata: folderPreparationMetadata(
                    for: url,
                    outcome: outcome,
                    encoderVersion: activeEncoderVersion
                ).merging(
                    ["pending_change_count": "\(pendingObservedFileChanges)"],
                    uniquingKeysWith: { _, new in new }
                )
            )
        } catch {
            libraryStateStore.folderPreparationProgress = []
            if let database = try? ensureDatabase(),
               let currentState = try? folderSearchState(for: url.path(percentEncoded: false), database: database) {
                applyLibrarySearchState(currentState, searchIssue: searchIssueMessage(for: error))
            }
            statusStore.status = libraryStateStore.selectedFolder == nil ? .readyWithoutFolder : .ready
            recordRuntimeLog(
                searchIssueMessage(for: error),
                level: .error,
                category: "library",
                metadata: [
                    "details": String(describing: error),
                    "encoder_version": activeEncoderVersion,
                    "folder": url.path(percentEncoded: false),
                    "pending_change_count": "\(pendingObservedFileChanges)",
                ]
            )
        }
    }

    private func performDeleteItems(withIDs itemIDs: Set<Int64>) async {
        guard itemIDs.isEmpty == false else {
            return
        }

        let shouldClosePreview = workspaceStateStore.previewAssetID.map(itemIDs.contains) ?? false
        let selectedItems = workspaceStateStore.results.filter { itemIDs.contains($0.id) }
        do {
            var removedPaths: [String] = []
            for item in selectedItems {
                try FileManager.default.trashItem(at: URL(filePath: item.absolutePath), resultingItemURL: nil)
                removedPaths.append(item.absolutePath)
            }

            if removedPaths.isEmpty == false {
                let database = try ensureDatabase()
                try database.markFileInstancesMissing(absolutePaths: removedPaths)
                if let selectedFolder = libraryStateStore.selectedFolder {
                    applyLibrarySearchState(
                        try folderSearchState(
                            for: selectedFolder.path(percentEncoded: false),
                            database: database
                        )
                    )
                }
            }

            activeSearchView = activeSearchView?.removing(recordIDs: itemIDs)

            if shouldClosePreview {
                workspaceStateStore.closePreview()
            }

            if let selectedFolder = libraryStateStore.selectedFolder {
                try? await refreshVisibleResults(for: selectedFolder)
            } else {
                workspaceStateStore.results.removeAll { itemIDs.contains($0.id) }
                if workspaceStateStore.previewItem == nil {
                    workspaceStateStore.closePreview()
                }
                workspaceStateStore.clearSelection()
            }
        } catch {
            if shouldClosePreview {
                workspaceStateStore.closePreview()
            }
            workspaceStateStore.clearSelection()
        }
    }

    private func replaceFolderProgressLocally(_ progress: FolderPreparationProgress) {
        if let index = libraryStateStore.folderPreparationProgress.lastIndex(where: { $0.step == progress.step }) {
            libraryStateStore.folderPreparationProgress[index] = progress
        } else {
            libraryStateStore.folderPreparationProgress.append(progress)
        }
    }

    private func replaceInstallProgressLocally(_ progress: InstallProgress) {
        if let index = installationStateStore.progress.lastIndex(where: { $0.step == progress.step }) {
            installationStateStore.progress[index] = progress
        } else {
            installationStateStore.progress.append(progress)
        }
    }

    private func restoreFolderAccess(_ previousSelectedFolder: URL?) {
        if let previousSelectedFolder {
            folderAccessSession.activate(previousSelectedFolder)
        } else {
            folderAccessSession.deactivate()
        }
    }

    private func replacePrivateAdaptationProgress(_ progress: PrivateAdaptationProgress) {
        if let index = libraryStateStore.privateAdaptationProgress.lastIndex(where: { $0.step == progress.step }) {
            libraryStateStore.privateAdaptationProgress[index] = progress
        } else {
            libraryStateStore.privateAdaptationProgress.append(progress)
        }
    }

    private func configureEmbeddingService(for folderURL: URL?) async {
        guard let folderURL, let adaptationTrainer else {
            activeEmbeddingService = baseEmbeddingService
            activeEncoderVersion = baseEmbeddingService.encoderVersion
            libraryStateStore.activeEncoderVersion = activeEncoderVersion
            cachedVectorQuery = nil
            if let folderURL {
                recordRuntimeLog(
                    "Using the published semantic search encoder for this folder.",
                    category: "search",
                    metadata: [
                        "encoder_version": activeEncoderVersion,
                        "folder": folderURL.path(percentEncoded: false),
                    ]
                )
            }
            return
        }

        if let adaptation = try? await adaptationTrainer.existingArtifact(for: folderURL) {
            activeEmbeddingService = adaptation.embeddingService
            activeEncoderVersion = adaptation.artifact.encoderVersion
            recordRuntimeLog(
                "Using the folder-specific semantic search adaptation.",
                category: "search",
                metadata: [
                    "encoder_version": adaptation.artifact.encoderVersion,
                    "folder": folderURL.path(percentEncoded: false),
                ]
            )
        } else {
            activeEmbeddingService = baseEmbeddingService
            activeEncoderVersion = baseEmbeddingService.encoderVersion
            recordRuntimeLog(
                "Using the published semantic search encoder for this folder.",
                category: "search",
                metadata: [
                    "encoder_version": activeEncoderVersion,
                    "folder": folderURL.path(percentEncoded: false),
                ]
            )
        }
        libraryStateStore.activeEncoderVersion = activeEncoderVersion
        cachedVectorQuery = nil
    }

    private func prewarmActiveEmbeddingService() async {
        let service = activeEmbeddingService
        do {
            try await service.prepareForQueries()
        } catch {
            recordRuntimeLog(
                searchIssueMessage(for: error),
                level: .error,
                category: "search",
                metadata: [
                    "details": String(describing: error),
                    "encoder_version": activeEncoderVersion,
                ]
            )
        }
    }

    private func rebuildActiveEncoderIndex(for url: URL) async throws {
        let database = try ensureDatabase()
        let folderPath = url.path(percentEncoded: false)
        let startDate = Date()

        await MainActor.run {
            self.replacePrivateAdaptationProgress(
                PrivateAdaptationProgress(
                    step: .reindexLibrary,
                    message: "Refreshing the folder index",
                    stepProgress: 0.0,
                    overallProgress: 2.0 / 3.0,
                    elapsedSeconds: 0,
                    remainingSeconds: nil
                )
            )
        }

        let indexSummary = try await FolderIndexer(database: database).rebuildIndex(for: url)
        let missingEmbeddings = try database.missingEmbeddingWork(
            inFolderAbsolutePath: folderPath,
            encoderVersion: activeEncoderVersion
        )

        if missingEmbeddings.isEmpty {
            await MainActor.run {
                self.replacePrivateAdaptationProgress(
                    PrivateAdaptationProgress(
                        step: .reindexLibrary,
                        message: "Search index is already current",
                        stepProgress: 1.0,
                        overallProgress: 1.0,
                        elapsedSeconds: Int(Date().timeIntervalSince(startDate)),
                        remainingSeconds: 0
                    )
                )
            }
        } else {
            for (index, item) in missingEmbeddings.enumerated() {
                do {
                    let vector = try await activeEmbeddingService.encodeImage(at: URL(filePath: item.representativeAbsolutePath))
                    try database.upsertEmbedding(
                        assetID: item.assetID,
                        encoderVersion: activeEncoderVersion,
                        vector: vector
                    )
                } catch let error as GalleryEmbeddingError where error == .invalidImageData {
                    continue
                }
                let completed = Double(index + 1) / Double(missingEmbeddings.count)
                let elapsed = Int(Date().timeIntervalSince(startDate))
                let remaining = index >= 0 ? Int((Double(elapsed) / Double(index + 1)) * Double(missingEmbeddings.count - (index + 1))) : nil
                await MainActor.run {
                    self.replacePrivateAdaptationProgress(
                        PrivateAdaptationProgress(
                            step: .reindexLibrary,
                            message: "Rebuilt \(index + 1) of \(missingEmbeddings.count) embeddings",
                            stepProgress: completed,
                            overallProgress: (2.0 + completed) / 3.0,
                            elapsedSeconds: elapsed,
                            remainingSeconds: remaining
                        )
                    )
                }
            }
        }

        activeSearchView = try searchCoordinator(database: database).activeSearchView(folderAbsolutePath: indexSummary.folderPath)
        applyLibrarySearchState(try folderSearchState(for: indexSummary.folderPath, database: database))
        try await refreshVisibleResults(for: url)
    }

    private func beginFolderSelection(at url: URL) {
        resetFolderPreparationRuntimeLogState()
        recordRuntimeLog(
            "Preparing the selected folder for search.",
            category: "library",
            metadata: [
                "encoder_version": activeEncoderVersion,
                "folder": url.path(percentEncoded: false),
            ]
        )
        folderAccessSession.activate(url)
        libraryStateStore.selectedFolder = url
        libraryStateStore.folderPreparationProgress = FolderPreparationProgress.placeholders()
        libraryStateStore.privateAdaptationNotice = nil
        libraryStateStore.privateAdaptationProgress = []
        resetActiveSearchSession()
    }

    private func resetActiveSearchSession() {
        libraryStateStore.searchIssue = nil
        libraryStateStore.searchableImageCount = nil
        workspaceStateStore.results = []
        workspaceStateStore.clearSelection()
        workspaceStateStore.leaveSelectionMode()
        workspaceStateStore.closePreview()
        workspaceStateStore.pastedImageData = nil
        activeSearchView = nil
        cachedVectorQuery = nil
    }

    private func finishPreparedFolderSelection(at url: URL) {
        statusStore.status = .ready
        libraryStateStore.folderPreparationProgress = []
        pendingObservedFileChanges = 0
        activeViewNeedsRefresh = false
        startFolderObservation(for: url)
    }

    private func recordFolderSelectionFailure(
        _ error: Error,
        at url: URL,
        includeDetails: Bool = false
    ) {
        var metadata = [
            "encoder_version": activeEncoderVersion,
            "folder": url.path(percentEncoded: false),
        ]
        if includeDetails {
            metadata["details"] = String(describing: error)
        }

        recordRuntimeLog(
            searchIssueMessage(for: error),
            level: .error,
            category: "library",
            metadata: metadata
        )
    }

    private func finishFailedFolderSelection(
        at url: URL,
        previousSelectedFolder: URL?,
        searchIssue: String? = nil
    ) async {
        if await recoverPreparedFolderSelection(at: url, searchIssue: searchIssue) == false {
            restorePreviousFolderSelection(previousSelectedFolder)
        }
    }

    private func restorePreviousFolderSelection(_ previousSelectedFolder: URL?) {
        restoreFolderAccess(previousSelectedFolder)
        libraryStateStore.selectedFolder = previousSelectedFolder
        startFolderObservation(for: previousSelectedFolder)
        statusStore.status = previousSelectedFolder == nil ? .readyWithoutFolder : .ready
    }

    private func vectorQuery(forText text: String) async throws -> [Double] {
        if let cachedVectorQuery,
           cachedVectorQuery.encoderVersion == activeEncoderVersion,
           cachedVectorQuery.source == .text(text) {
            return cachedVectorQuery.vector
        }

        let vector = try await activeEmbeddingService.encodeText(text)
        cachedVectorQuery = CachedVectorQuery(
            encoderVersion: activeEncoderVersion,
            source: .text(text),
            vector: vector
        )
        return vector
    }

    private func vectorQuery(forImageData imageData: Data) async throws -> [Double] {
        if let cachedVectorQuery,
           cachedVectorQuery.encoderVersion == activeEncoderVersion,
           cachedVectorQuery.source == .image(imageData) {
            return cachedVectorQuery.vector
        }

        let vector = try await activeEmbeddingService.encodeImage(data: imageData)
        cachedVectorQuery = CachedVectorQuery(
            encoderVersion: activeEncoderVersion,
            source: .image(imageData),
            vector: vector
        )
        return vector
    }
}

private struct FolderPreparationOutcome: Sendable {
    let folderPath: String
    let supportedImageCount: Int
    let reusedEmbeddingCount: Int
    let generatedEmbeddingCount: Int
    let skippedImageCount: Int
    let searchState: FolderSearchState
}

private struct CachedVectorQuery {
    let encoderVersion: String
    let source: VectorQuerySource
    let vector: [Double]
}

private struct FolderSearchState: Sendable {
    let supportedImageCount: Int
    let searchableImageCount: Int
}

private enum VectorQuerySource: Equatable {
    case text(String)
    case image(Data)
}

private func searchIssueMessage(for error: Error) -> String {
    switch error {
    case GalleryEmbeddingError.invalidModelArtifact:
        return "Semantic search model files could not be loaded from the app bundle."
    case GalleryEmbeddingError.invalidStage1Weights:
        return "Semantic search weights could not be loaded from the app bundle."
    case GalleryEmbeddingError.invalidImageData:
        return "Semantic search could not prepare embeddings for this folder."
    case GalleryEmbeddingError.invalidAdapterArtifact:
        return "Semantic search could not load the folder adaptation, so only the published encoder is available."
    default:
        return "Semantic search could not finish preparing embeddings for this folder."
    }
}

private func folderPreparationMetadata(
    for url: URL,
    outcome: FolderPreparationOutcome,
    encoderVersion: String
) -> [String: String] {
    [
        "encoder_version": encoderVersion,
        "folder": url.path(percentEncoded: false),
        "generated_embeddings": "\(outcome.generatedEmbeddingCount)",
        "reused_embeddings": "\(outcome.reusedEmbeddingCount)",
        "searchable_images": "\(outcome.searchState.searchableImageCount)",
        "skipped_images": "\(outcome.skippedImageCount)",
        "supported_images": "\(outcome.supportedImageCount)",
    ]
}

private func prepareFolderSelection(
    at url: URL,
    database: LibraryDatabase,
    embeddingService: any GalleryEmbeddingService,
    encoderVersion: String,
    stepDelayNanoseconds: UInt64,
    onProgress: @escaping @Sendable (FolderPreparationProgress) async -> Void
) async throws -> FolderPreparationOutcome {
    let totalSteps = 6.0

    let accessStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .requestFolderAccess,
            message: "Secure access is ready",
            stepProgress: 1.0,
            overallProgress: 1.0 / totalSteps,
            elapsedSeconds: Int(Date().timeIntervalSince(accessStart)),
            remainingSeconds: 0
        )
    )
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let scanStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Scanning supported images",
            stepProgress: 0,
            overallProgress: 1.0 / totalSteps,
            elapsedSeconds: 0,
            remainingSeconds: nil,
            recordedAt: Date()
        )
    )
    let indexSummary = try await FolderIndexer(database: database).rebuildIndex(for: url) { snapshot in
        let stepProgress: Double
        if snapshot.totalCount == 0 {
            stepProgress = 1.0
        } else {
            stepProgress = Double(snapshot.processedCount) / Double(snapshot.totalCount)
        }
        let elapsed = Int(Date().timeIntervalSince(scanStart))
        let remaining = snapshot.processedCount > 0 && snapshot.totalCount > snapshot.processedCount
            ? Int((Double(elapsed) / Double(snapshot.processedCount)) * Double(snapshot.totalCount - snapshot.processedCount))
            : 0

        await onProgress(
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Scanned \(snapshot.processedCount) of \(snapshot.totalCount) images · \(snapshot.uniqueAssetCount) unique assets",
                stepProgress: stepProgress,
                overallProgress: (1.0 + stepProgress) / totalSteps,
                elapsedSeconds: elapsed,
                remainingSeconds: remaining,
                recordedAt: Date()
            )
        )
    }

    if indexSummary.fileCount == 0 {
        await onProgress(
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Scanned 0 of 0 images",
                stepProgress: 1.0,
                overallProgress: 2.0 / totalSteps,
                elapsedSeconds: Int(Date().timeIntervalSince(scanStart)),
                remainingSeconds: 0,
                recordedAt: Date()
            )
        )
    } else {
        await onProgress(
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Scanned \(indexSummary.fileCount) of \(indexSummary.fileCount) images · \(indexSummary.uniqueAssetCount) unique assets",
                stepProgress: 1.0,
                overallProgress: 2.0 / totalSteps,
                elapsedSeconds: Int(Date().timeIntervalSince(scanStart)),
                remainingSeconds: 0,
                recordedAt: Date()
            )
        )
    }
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let missingEmbeddings = try database.missingEmbeddingWork(
        inFolderAbsolutePath: indexSummary.folderPath,
        encoderVersion: encoderVersion
    )
    let reusedCount = max(0, indexSummary.uniqueAssetCount - missingEmbeddings.count)
    let reuseStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .reuseExistingEmbeddings,
            message: "Reused \(reusedCount) embeddings, \(missingEmbeddings.count) still needed",
            stepProgress: 1.0,
            overallProgress: 3.0 / totalSteps,
            elapsedSeconds: Int(Date().timeIntervalSince(reuseStart)),
            remainingSeconds: 0,
            recordedAt: Date()
        )
    )
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let embeddingStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .generateEmbeddings,
            message: missingEmbeddings.isEmpty ? "All embeddings are already available" : "Generating search embeddings",
            stepProgress: 0,
            overallProgress: 3.0 / totalSteps,
            elapsedSeconds: 0,
            remainingSeconds: nil,
            recordedAt: Date()
        )
    )
    var skippedImageCount = 0
    if missingEmbeddings.isEmpty {
        await onProgress(
            FolderPreparationProgress(
                step: .generateEmbeddings,
                message: "Generated 0 of 0 embeddings",
                stepProgress: 1.0,
                overallProgress: 4.0 / totalSteps,
                elapsedSeconds: Int(Date().timeIntervalSince(embeddingStart)),
                remainingSeconds: 0,
                recordedAt: Date()
            )
        )
    } else {
        for (index, item) in missingEmbeddings.enumerated() {
            do {
                let vector = try await embeddingService.encodeImage(at: URL(filePath: item.representativeAbsolutePath))
                try database.upsertEmbedding(
                    assetID: item.assetID,
                    encoderVersion: encoderVersion,
                    vector: vector
                )
            } catch let error as GalleryEmbeddingError where error == .invalidImageData {
                skippedImageCount += 1
            }

            let completed = Double(index + 1) / Double(missingEmbeddings.count)
            let elapsed = Int(Date().timeIntervalSince(embeddingStart))
            let remaining = index >= 0
                ? Int((Double(elapsed) / Double(index + 1)) * Double(missingEmbeddings.count - (index + 1)))
                : nil
            let statusText = skippedImageCount > 0
                ? "Prepared \(index + 1) of \(missingEmbeddings.count) images · \(skippedImageCount) skipped"
                : "Prepared \(index + 1) of \(missingEmbeddings.count) embeddings"
            await onProgress(
                FolderPreparationProgress(
                    step: .generateEmbeddings,
                    message: statusText,
                    stepProgress: completed,
                    overallProgress: (3.0 + completed) / totalSteps,
                    elapsedSeconds: elapsed,
                    remainingSeconds: remaining,
                    recordedAt: Date()
                )
            )
        }
    }
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let supportedCount = try database.visibleFileInstances(
        inFolderAbsolutePath: indexSummary.folderPath,
        limit: Int.max
    ).count
    let searchableCount = try database.embeddedFileCount(
        inFolderAbsolutePath: indexSummary.folderPath,
        encoderVersion: encoderVersion
    )
    let buildStart = Date()
    let buildMessage: String
    if searchableCount < supportedCount {
        let noun = supportedCount == 1 ? "image" : "images"
        buildMessage = "\(searchableCount) of \(supportedCount) \(noun) ready for search"
    } else {
        let noun = searchableCount == 1 ? "image" : "images"
        buildMessage = "\(searchableCount) \(noun) ready for search"
    }
    await onProgress(
        FolderPreparationProgress(
            step: .buildSearchView,
            message: buildMessage,
            stepProgress: 1.0,
            overallProgress: 5.0 / totalSteps,
            elapsedSeconds: Int(Date().timeIntervalSince(buildStart)),
            remainingSeconds: 0,
            recordedAt: Date()
        )
    )
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let finalizeStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .finalizeFolderSelection,
            message: "Folder is ready",
            stepProgress: 1.0,
            overallProgress: 1.0,
            elapsedSeconds: Int(Date().timeIntervalSince(finalizeStart)),
            remainingSeconds: 0,
            recordedAt: Date()
        )
    )

    return FolderPreparationOutcome(
        folderPath: indexSummary.folderPath,
        supportedImageCount: supportedCount,
        reusedEmbeddingCount: reusedCount,
        generatedEmbeddingCount: max(0, searchableCount - reusedCount),
        skippedImageCount: skippedImageCount,
        searchState: FolderSearchState(
            supportedImageCount: supportedCount,
            searchableImageCount: searchableCount
        )
    )
}

private func pauseBetweenFolderSteps(_ stepDelayNanoseconds: UInt64) async throws {
    if stepDelayNanoseconds > 0 {
        try await Task.sleep(nanoseconds: stepDelayNanoseconds)
    } else {
        await Task.yield()
    }
}
