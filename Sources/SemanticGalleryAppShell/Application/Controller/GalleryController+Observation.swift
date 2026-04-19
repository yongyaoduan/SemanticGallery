import Foundation
import SemanticGalleryCore
import SemanticGalleryInstall
import SemanticGalleryPersistence

extension GalleryController {
    func shouldReindexRememberedFolder(_ url: URL) -> Bool {
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

    func startFolderObservation(for url: URL?) {
        observedFolderSyncTask?.cancel()
        observedFolderSyncTask = nil
        pendingObservedFileChanges = 0
        folderSearchIndexNeedsRefresh = false
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

    func recordObservedFolderChanges(_ changeCount: Int) async {
        guard let selectedFolder = libraryState.selectedFolder else {
            return
        }

        pendingObservedFileChanges += max(changeCount, 1)
        folderSearchIndexNeedsRefresh = true

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

    func performObservedFolderSync(for url: URL) async {
        defer { observedFolderSyncTask = nil }

        guard let selectedFolder = libraryState.selectedFolder,
              selectedFolder.standardizedFileURL == url.standardizedFileURL else {
            return
        }

        statusState.status = .indexing
        libraryState.folderPreparationProgress = FolderPreparationProgress.placeholders()
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
                currentViewExists: folderSearchIndex != nil,
                pendingChangeCount: pendingObservedFileChanges,
                visibleImageCount: outcome.supportedImageCount
            ) {
                try refreshFolderSearchIndex(for: url)
            }
            try await refreshVisibleResults(for: url)
            libraryState.folderPreparationProgress = []
            statusState.status = .ready
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
            libraryState.folderPreparationProgress = []
            if let database = try? ensureDatabase(),
               let currentState = try? folderSearchState(for: url.path(percentEncoded: false), database: database) {
                applyLibrarySearchState(currentState, searchIssue: searchIssueMessage(for: error))
            }
            statusState.status = libraryState.selectedFolder == nil ? .readyWithoutFolder : .ready
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

    func recordRuntimeLog(
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

    func resetFolderPreparationRuntimeLogState() {
        lastLoggedFolderPreparationStep = nil
        lastLoggedFolderPreparationAt = .distantPast
    }

    func recordFolderPreparationProgress(_ progress: FolderPreparationProgress, for url: URL) {
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

    func folderPreparationStepIdentifier(_ step: FolderPreparationStep) -> String {
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

    func replaceInstallProgressLocally(_ progress: InstallProgress) {
        if let index = installationState.progress.lastIndex(where: { $0.step == progress.step }) {
            installationState.progress[index] = progress
        } else {
            installationState.progress.append(progress)
        }
    }
}
