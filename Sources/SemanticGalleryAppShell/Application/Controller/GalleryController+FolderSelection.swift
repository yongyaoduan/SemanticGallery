import Foundation
import SemanticGalleryCore
import SemanticGalleryIndex
import SemanticGalleryML
import SemanticGalleryPersistence
import SemanticGallerySearch

extension GalleryController {
    public func selectFolder(at url: URL) async {
        let previousSelectedFolder = libraryState.selectedFolder
        startFolderObservation(for: nil)
        statusState.status = .indexing
        ensureRuntimeDirectoriesExist()
        folderAccessSession.activate(url)
        libraryState.selectedFolder = url
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
                    if self.libraryState.folderPreparationProgress.contains(where: { $0.step == progress.step }) {
                        self.replaceFolderProgressLocally(progress)
                    } else {
                        self.libraryState.folderPreparationProgress.append(progress)
                    }
                    self.recordFolderPreparationProgress(progress, for: url)
                }
            }
            guard selectedFolderMatches(url) else {
                return
            }
            folderSearchIndex = try? searchCoordinator(database: database).folderSearchIndex(folderAbsolutePath: outcome.folderPath)
            try await refreshVisibleResults(for: url)
            guard selectedFolderMatches(url) else {
                return
            }
            applyLibrarySearchState(outcome.searchState)
            libraryState.activeEncoderVersion = activeEncoderVersion
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
            guard selectedFolderMatches(url) else {
                return
            }
            libraryState.folderPreparationProgress = []
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
                libraryState.activeEncoderVersion = activeEncoderVersion
                cachedVectorQuery = nil
                libraryState.privateAdaptationNotice = "The local adaptation could not be loaded, so SemanticGallery returned to the published encoder."
                await finishFailedFolderSelection(at: url, previousSelectedFolder: previousSelectedFolder)
            case .invalidImageData:
                await finishFailedFolderSelection(
                    at: url,
                    previousSelectedFolder: previousSelectedFolder,
                    searchIssue: searchIssueMessage(for: error)
                )
            }
        } catch {
            guard selectedFolderMatches(url) else {
                return
            }
            libraryState.folderPreparationProgress = []
            recordFolderSelectionFailure(error, at: url, includeDetails: true)
            await finishFailedFolderSelection(
                at: url,
                previousSelectedFolder: previousSelectedFolder,
                searchIssue: searchIssueMessage(for: error)
            )
        }
    }

    func rebuildWorkspace(for url: URL, shouldReindex: Bool = true) async throws {
        let database = try ensureDatabase()
        if shouldReindex {
            let summary = try await FolderIndexer(database: database).rebuildIndex(for: url)
            guard selectedFolderMatches(url) else {
                return
            }
            try await populateMissingEmbeddings(
                inFolderAbsolutePath: summary.folderPath,
                database: database
            )
            guard selectedFolderMatches(url) else {
                return
            }
            applyLibrarySearchState(try folderSearchState(for: summary.folderPath, database: database))
        } else {
            guard selectedFolderMatches(url) else {
                return
            }
            applyLibrarySearchState(try folderSearchState(for: url.path(percentEncoded: false), database: database))
        }

        guard selectedFolderMatches(url) else {
            return
        }
        try refreshFolderSearchIndex(for: url)
        usageState.pastedImageData = nil
        try await refreshVisibleResults(for: url)
    }

    func populateMissingEmbeddings(
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

    func ensureFolderSearchIndexIsCurrent(for url: URL) throws {
        if folderSearchIndex == nil || folderSearchIndexNeedsRefresh {
            try refreshFolderSearchIndex(for: url)
        }
    }

    func refreshFolderSearchIndex(for url: URL) throws {
        let coordinator = searchCoordinator(database: try ensureDatabase())
        folderSearchIndex = try coordinator.folderSearchIndex(folderAbsolutePath: url.path(percentEncoded: false))
        pendingObservedFileChanges = 0
        folderSearchIndexNeedsRefresh = false
    }

    func beginFolderSelection(at url: URL) {
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
        libraryState.selectedFolder = url
        libraryState.folderPreparationProgress = FolderPreparationProgress.placeholders()
        libraryState.privateAdaptationNotice = nil
        libraryState.privateAdaptationProgress = []
        resetActiveSearchSession()
    }

    func finishPreparedFolderSelection(at url: URL) {
        statusState.status = .ready
        libraryState.folderPreparationProgress = []
        pendingObservedFileChanges = 0
        folderSearchIndexNeedsRefresh = false
        startFolderObservation(for: url)
    }

    func recordFolderSelectionFailure(
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

    func finishFailedFolderSelection(
        at url: URL,
        previousSelectedFolder: URL?,
        searchIssue: String? = nil
    ) async {
        if await recoverPreparedFolderSelection(at: url, searchIssue: searchIssue) == false {
            restorePreviousFolderSelection(previousSelectedFolder)
        }
    }

    func recoverPreparedFolderSelection(
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
        libraryState.selectedFolder = url
        let coordinator = searchCoordinator(database: database)
        applyLibrarySearchState(
            (try? folderSearchState(for: folderPath, database: database))
                ?? FolderSearchState(supportedImageCount: visibleCount, searchableImageCount: 0),
            searchIssue: searchIssue
        )
        libraryState.activeEncoderVersion = activeEncoderVersion
        cachedVectorQuery = nil
        folderSearchIndex = try? coordinator.folderSearchIndex(folderAbsolutePath: folderPath)

        do {
            try await refreshVisibleResults(for: url)
        } catch {
            let lexicalQuery = usageState.queryText.trimmingCharacters(in: .whitespacesAndNewlines)
            usageState.results = (try? coordinator.search(
                folderAbsolutePath: folderPath,
                query: lexicalQuery,
                limit: usageState.resultLimit
            )) ?? []
            usageState.clearSelection()
            if usageState.previewItem == nil {
                usageState.closePreview()
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
                "searchable_images": "\(libraryState.searchableImageCount ?? 0)",
                "supported_images": "\(libraryState.supportedImageCount ?? visibleCount)",
            ]
        )
        return true
    }

    func restorePreviousFolderSelection(_ previousSelectedFolder: URL?) {
        restoreFolderAccess(previousSelectedFolder)
        libraryState.selectedFolder = previousSelectedFolder
        startFolderObservation(for: previousSelectedFolder)
        statusState.status = previousSelectedFolder == nil ? .readyWithoutFolder : .ready
    }

    func restoreFolderAccess(_ previousSelectedFolder: URL?) {
        if let previousSelectedFolder {
            folderAccessSession.activate(previousSelectedFolder)
        } else {
            folderAccessSession.deactivate()
        }
    }

    func selectedFolderMatches(_ url: URL) -> Bool {
        libraryState.selectedFolder?.standardizedFileURL == url.standardizedFileURL
    }

    func replaceFolderProgressLocally(_ progress: FolderPreparationProgress) {
        if let index = libraryState.folderPreparationProgress.lastIndex(where: { $0.step == progress.step }) {
            libraryState.folderPreparationProgress[index] = progress
        } else {
            libraryState.folderPreparationProgress.append(progress)
        }
    }
}
