import Foundation
import SemanticGalleryPersistence
import SemanticGallerySearch

extension GalleryController {
    public func runSearch() {
        guard let selectedFolder = libraryState.selectedFolder else {
            usageState.results = []
            usageState.clearSelection()
            if usageState.previewItem == nil {
                usageState.closePreview()
            }
            return
        }

        Task {
            do {
                try await refreshVisibleResults(for: selectedFolder)
            } catch {
                await MainActor.run {
                    self.usageState.results = []
                    self.usageState.clearSelection()
                }
            }
        }
    }

    public func toggleSelection(resultID: Int64) {
        usageState.toggleSelection(resultID: resultID)
    }

    public func enterSelectionMode() {
        usageState.enterSelectionMode()
    }

    public func leaveSelectionMode() {
        usageState.leaveSelectionMode()
    }

    public func selectAllVisibleResults() {
        usageState.selectAllVisible()
    }

    public func clearSelection() {
        usageState.clearSelection()
    }

    public func openPreview(resultID: Int64) {
        usageState.openPreview(resultID: resultID)
    }

    public func closePreview() {
        usageState.closePreview()
    }

    public func togglePreviewMetadata() {
        usageState.togglePreviewMetadata()
    }

    public func searchSimilarToPreviewItem() {
        guard
            let previewItem = usageState.previewItem,
            let selectedFolder = libraryState.selectedFolder,
            let imageData = try? Data(contentsOf: URL(filePath: previewItem.absolutePath))
        else {
            return
        }

        usageState.queryText = ""
        usageState.pastedImageData = imageData
        usageState.closePreview()

        Task {
            do {
                try await refreshVisibleResults(for: selectedFolder)
            } catch {
                await MainActor.run {
                    self.usageState.results = []
                    self.usageState.clearSelection()
                }
            }
        }
    }

    public func showNextPreviewItem() {
        usageState.showNextPreviewItem()
    }

    public func showPreviousPreviewItem() {
        usageState.showPreviousPreviewItem()
    }

    public func deletePreviewResult() {
        guard let previewResultID = usageState.previewResultID else {
            return
        }

        Task {
            await performDeleteResults(withIDs: [previewResultID])
        }
    }

    public func deleteSelectedResults() {
        Task {
            await performDeleteResults(withIDs: usageState.selectedResultIDs)
        }
    }

    func refreshVisibleResults(for url: URL) async throws {
        guard selectedFolderMatches(url) else {
            return
        }
        let database = try ensureDatabase()
        let coordinator = SearchCoordinator(
            database: database,
            encoderVersion: activeEncoderVersion
        )
        usageState.isSearching = true
        let previousStatus = statusState.status
        if previousStatus != .indexing && previousStatus != .training {
            statusState.status = .searching
        }
        defer {
            usageState.isSearching = false
            if self.statusState.status == .searching {
                self.statusState.status = previousStatus == .searching
                    ? (self.libraryState.selectedFolder == nil ? .readyWithoutFolder : .ready)
                    : previousStatus
            }
        }

        let folderPath = url.path(percentEncoded: false)
        let trimmedQuery = usageState.queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let queryLimit = usageState.resultLimit

        if let pastedImageData = usageState.pastedImageData {
            let queryVector = try await vectorQuery(forImageData: pastedImageData)
            guard selectedFolderMatches(url) else {
                return
            }
            try ensureFolderSearchIndexIsCurrent(for: url)
            guard selectedFolderMatches(url) else {
                return
            }

            if let folderSearchIndex, folderSearchIndex.items.isEmpty == false {
                usageState.results = coordinator.vectorSearch(
                    searchIndex: folderSearchIndex,
                    queryVector: queryVector,
                    limit: queryLimit
                )
            } else {
                recordSemanticSearchUnavailable(
                    for: url,
                    queryType: "image"
                )
                usageState.results = []
            }
        } else if trimmedQuery.isEmpty == false {
            let lexicalMatches = try coordinator.search(
                folderAbsolutePath: folderPath,
                query: trimmedQuery,
                limit: queryLimit
            )
            guard selectedFolderMatches(url) else {
                return
            }

            if lexicalMatches.isEmpty == false {
                usageState.results = lexicalMatches
            } else {
                let queryVector = try await vectorQuery(forText: trimmedQuery)
                guard selectedFolderMatches(url) else {
                    return
                }
                try ensureFolderSearchIndexIsCurrent(for: url)
                guard selectedFolderMatches(url) else {
                    return
                }

                if let folderSearchIndex, folderSearchIndex.items.isEmpty == false {
                    usageState.results = coordinator.vectorSearch(
                        searchIndex: folderSearchIndex,
                        queryVector: queryVector,
                        limit: queryLimit
                    )
                } else {
                    recordSemanticSearchUnavailable(
                        for: url,
                        queryType: "text"
                    )
                    usageState.results = []
                }
            }
        } else {
            usageState.results = try coordinator.search(
                folderAbsolutePath: folderPath,
                query: trimmedQuery,
                limit: queryLimit
            )
        }

        if usageState.previewItem == nil {
            usageState.closePreview()
        }
        usageState.clearSelection()
    }

    func folderSearchState(
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

    func applyLibrarySearchState(
        _ state: FolderSearchState,
        searchIssue: String? = nil
    ) {
        libraryState.supportedImageCount = state.supportedImageCount
        libraryState.searchableImageCount = state.searchableImageCount

        if let searchIssue, state.searchableImageCount < state.supportedImageCount {
            libraryState.searchIssue = searchIssue
        } else if state.supportedImageCount > 0, state.searchableImageCount == 0 {
            libraryState.searchIssue = "Semantic search could not prepare embeddings for this folder."
        } else {
            libraryState.searchIssue = nil
        }
    }

    func recordSemanticSearchUnavailable(for url: URL, queryType: String) {
        recordRuntimeLog(
            "Semantic search could not run because no searchable embeddings are available.",
            level: .warning,
            category: "search",
            metadata: [
                "encoder_version": activeEncoderVersion,
                "folder": url.path(percentEncoded: false),
                "query_type": queryType,
                "searchable_images": "\(libraryState.searchableImageCount ?? 0)",
                "supported_images": "\(libraryState.supportedImageCount ?? 0)",
            ]
        )
    }

    func resetActiveSearchSession() {
        libraryState.searchIssue = nil
        libraryState.searchableImageCount = nil
        usageState.results = []
        usageState.clearSelection()
        usageState.leaveSelectionMode()
        usageState.closePreview()
        usageState.pastedImageData = nil
        folderSearchIndex = nil
        cachedVectorQuery = nil
    }

    func vectorQuery(forText text: String) async throws -> [Double] {
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

    func vectorQuery(forImageData imageData: Data) async throws -> [Double] {
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

    func performDeleteResults(withIDs resultIDs: Set<Int64>) async {
        guard resultIDs.isEmpty == false else {
            return
        }

        let shouldClosePreview = usageState.previewResultID.map(resultIDs.contains) ?? false
        let selectedItems = usageState.results.filter { resultIDs.contains($0.id) }
        do {
            var removedPaths: [String] = []
            for item in selectedItems {
                try FileManager.default.trashItem(at: URL(filePath: item.absolutePath), resultingItemURL: nil)
                removedPaths.append(item.absolutePath)
            }

            if removedPaths.isEmpty == false {
                let database = try ensureDatabase()
                try database.markFileInstancesMissing(absolutePaths: removedPaths)
                if let selectedFolder = libraryState.selectedFolder {
                    applyLibrarySearchState(
                        try folderSearchState(
                            for: selectedFolder.path(percentEncoded: false),
                            database: database
                        )
                    )
                }
            }

            folderSearchIndex = folderSearchIndex?.removing(resultIDs: resultIDs)

            if shouldClosePreview {
                usageState.closePreview()
            }

            if let selectedFolder = libraryState.selectedFolder {
                try? await refreshVisibleResults(for: selectedFolder)
            } else {
                usageState.results.removeAll { resultIDs.contains($0.id) }
                if usageState.previewItem == nil {
                    usageState.closePreview()
                }
                usageState.clearSelection()
            }
        } catch {
            if shouldClosePreview {
                usageState.closePreview()
            }
            usageState.clearSelection()
        }
    }
}
