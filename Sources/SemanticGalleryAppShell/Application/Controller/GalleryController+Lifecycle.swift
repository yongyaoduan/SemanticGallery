import AppKit
import Foundation
import SemanticGalleryCore
import SemanticGalleryInstall
import SemanticGalleryML
import SemanticGalleryPersistence
import SemanticGallerySearch

extension GalleryController {
    public func startInstallation() {
        guard statusState.status != .installing else {
            return
        }

        let installStepDelayNanoseconds = runtimeOptions.installStepDelayNanoseconds
        statusState.status = .installing
        installationState.progress = InstallProgress.placeholders()

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
                    self.installationState.progress = progress
                    self.statusState.status = .installComplete
                }
                await self.prewarmActiveEmbeddingService()
            } catch {
                await MainActor.run {
                    self.installationState.progress = []
                    self.statusState.status = .installFailed("Installation could not finish.")
                }
            }
        }
    }

    public func enterWorkspace() {
        statusState.status = libraryState.selectedFolder == nil ? .readyWithoutFolder : .ready
    }

    public func chooseFolder() {
        guard let url = folderPickerCoordinator.pickFolder() else {
            return
        }

        Task {
            await selectFolder(at: url)
        }
    }

    public func startUninstall() {
        statusState.status = .uninstalling
        let selectedFolder = libraryState.selectedFolder
        startFolderObservation(for: nil)

        libraryState.selectedFolder = nil
        libraryState.folderPreparationProgress = []
        installationState.progress = []
        usageState.results = []
        usageState.pastedImageData = nil
        usageState.closePreview()
        usageState.leaveSelectionMode()
        usageState.clearSelection()
        libraryState.privateAdaptationProgress = []
        libraryState.privateAdaptationNotice = nil
        libraryState.supportedImageCount = nil
        libraryState.searchableImageCount = nil
        libraryState.activeEncoderVersion = nil
        libraryState.searchIssue = nil
        folderSearchIndex = nil
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
                statusState.status = .readyWithoutFolder
            }
        } catch {
            statusState.status = .installFailed("Unable to remove SemanticGallery data.")
        }
    }

    func bootstrap() {
        do {
            if let selectedFolder = try bookmarkStore.loadBookmark() {
                folderAccessSession.activate(selectedFolder)
                libraryState.selectedFolder = selectedFolder
                libraryState.activeEncoderVersion = activeEncoderVersion
                Task {
                    await self.configureEmbeddingService(for: selectedFolder)
                    guard self.selectedFolderMatches(selectedFolder) else {
                        return
                    }
                    let shouldReindex = self.shouldReindexRememberedFolder(selectedFolder)
                    await MainActor.run {
                        guard self.selectedFolderMatches(selectedFolder) else {
                            return
                        }
                        self.statusState.status = shouldReindex ? .indexing : .ready
                    }
                    do {
                        try await self.rebuildWorkspace(for: selectedFolder, shouldReindex: shouldReindex)
                        guard self.selectedFolderMatches(selectedFolder) else {
                            return
                        }
                        await MainActor.run {
                            self.statusState.status = .ready
                        }
                        self.startFolderObservation(for: selectedFolder)
                    } catch {
                        guard self.selectedFolderMatches(selectedFolder) else {
                            return
                        }
                        await MainActor.run {
                            self.folderAccessSession.deactivate()
                            self.libraryState.selectedFolder = nil
                            self.libraryState.supportedImageCount = nil
                            self.libraryState.searchableImageCount = nil
                            self.usageState.results = []
                            self.usageState.clearSelection()
                            self.usageState.leaveSelectionMode()
                            self.usageState.closePreview()
                            self.usageState.pastedImageData = nil
                            self.libraryState.folderPreparationProgress = []
                            self.libraryState.searchIssue = nil
                            self.statusState.status = .readyWithoutFolder
                        }
                        self.bookmarkStore.clear()
                        self.startFolderObservation(for: nil)
                    }
                    if self.selectedFolderMatches(selectedFolder) {
                        await self.prewarmActiveEmbeddingService()
                    }
                }
            } else {
                startFolderObservation(for: nil)
                statusState.status = .readyWithoutFolder
                Task {
                    await self.prewarmActiveEmbeddingService()
                }
            }
        } catch {
            startFolderObservation(for: nil)
            statusState.status = .readyWithoutFolder
        }
    }

    func ensureDatabase() throws -> LibraryDatabase {
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

    func searchCoordinator(database: LibraryDatabase) -> SearchCoordinator {
        SearchCoordinator(
            database: database,
            encoderVersion: activeEncoderVersion
        )
    }

    func ensureRuntimeDirectoriesExist() {
        RuntimeDirectories.ensureExists(for: runtimeOptions.paths)
    }

    func configureEmbeddingService(for folderURL: URL?) async {
        guard let folderURL, let adaptationTrainer else {
            activeEmbeddingService = baseEmbeddingService
            activeEncoderVersion = baseEmbeddingService.encoderVersion
            libraryState.activeEncoderVersion = activeEncoderVersion
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
        libraryState.activeEncoderVersion = activeEncoderVersion
        cachedVectorQuery = nil
    }

    func prewarmActiveEmbeddingService() async {
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
}
