import Foundation
import Testing
@testable import SemanticGalleryAppShell
@testable import SemanticGalleryCore
@testable import SemanticGalleryIndex
@testable import SemanticGalleryInstall
@testable import SemanticGalleryML
@testable import SemanticGalleryPersistence
@Suite(.serialized)
struct SemanticGalleryControllerTests {
@Test
func controllerStartsReadyWithoutFolderWhenInstallIsComplete() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    let status = await MainActor.run { controller.statusStore.status }
    #expect(status == .readyWithoutFolder)
    try await waitForCondition("base embedding service prewarm from ready without folder") {
        await embeddingService.prepareCallCount() > 0
    }
}
@Test
func controllerInitializationCreatesTheApplicationSupportRootWhenItIsMissing() async throws {
    let baseRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: baseRoot) }
    let paths = AppPaths(root: baseRoot)
    try? FileManager.default.removeItem(at: paths.supportRoot)
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    _ = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: RecordingEmbeddingService(imageVectorsByBytes: [:], textVectors: [:])
        )
    }
    #expect(FileManager.default.fileExists(atPath: paths.supportRoot.path(percentEncoded: false)))
}
@Test
func controllerSelectingFolderBuildsWorkspaceResults() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    try Data("ignore".utf8).write(to: folder.appending(path: "notes.txt"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let status = await MainActor.run { controller.statusStore.status }
    let selectedFolder = await MainActor.run { controller.libraryStateStore.selectedFolder }
    let resultCount = await MainActor.run { controller.workspaceStateStore.results.count }
    let resultNames = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    let folderPreparationProgress = await MainActor.run { controller.libraryStateStore.folderPreparationProgress }
    let searchableImageCount = await MainActor.run { controller.libraryStateStore.searchableImageCount }
    let searchIssue = await MainActor.run { controller.libraryStateStore.searchIssue }
    #expect(status == .ready)
    #expect(selectedFolder == folder)
    #expect(resultCount == 2)
    #expect(Set(resultNames) == Set(["front.jpg", "back.png"]))
    #expect(folderPreparationProgress.isEmpty)
    #expect(searchableImageCount == 2)
    #expect(searchIssue == nil)
}
@Test
func controllerKeepsVisibleResultsWhenEmbeddingGenerationSkipsImages() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:],
        invalidImageBytes: ["front", "back"]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let status = await MainActor.run { controller.statusStore.status }
    let resultNames = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    let supportedImageCount = await MainActor.run { controller.libraryStateStore.supportedImageCount }
    let searchableImageCount = await MainActor.run { controller.libraryStateStore.searchableImageCount }
    let searchIssue = await MainActor.run { controller.libraryStateStore.searchIssue }
    let database = try LibraryDatabase.open(at: paths.databaseURL)
    try database.migrate()
    let visible = try database.visibleFileInstances(inFolderAbsolutePath: folder.path(percentEncoded: false), limit: 10)
    let embedded = try database.embeddedFiles(inFolderAbsolutePath: folder.path(percentEncoded: false), encoderVersion: "stage1")
    #expect(status == .ready)
    #expect(Set(resultNames) == Set(["front.jpg", "back.png"]))
    #expect(supportedImageCount == 2)
    #expect(searchableImageCount == 0)
    #expect(searchIssue == "Semantic search could not prepare embeddings for this folder.")
    #expect(visible.count == 2)
    #expect(embedded.isEmpty)
}
@Test
func controllerDeletingSelectionMovesFilesOutOfWorkspaceResults() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let removedPath = try #require(
        await MainActor.run { controller.workspaceStateStore.results.first?.absolutePath }
    )
    let removedAssetID = try #require(
        await MainActor.run { controller.workspaceStateStore.results.first?.assetID }
    )
    await MainActor.run {
        controller.toggleSelection(assetID: removedAssetID)
        controller.deleteSelection()
    }
    try await waitForCondition("workspace results to refresh after delete") {
        await MainActor.run { controller.workspaceStateStore.results.count == 1 }
    }
    let database = try LibraryDatabase.open(at: paths.databaseURL)
    try database.migrate()
    let visible = try database.visibleFileInstances(inFolderAbsolutePath: folder.path(percentEncoded: false), limit: 10)
    #expect(FileManager.default.fileExists(atPath: removedPath) == false)
    #expect(visible.count == 1)
}
@Test
func controllerPersistsTheChosenFolderInsideApplicationSupportAndForgetsItWhenSupportRootIsRemoved() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    var firstController: SemanticGalleryController? = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: RecordingEmbeddingService(imageVectorsByBytes: [:], textVectors: [:])
        )
    }
    await firstController?.selectFolder(at: folder)
    let bookmarkFileURL = paths.supportRoot.appending(path: "selected-folder.bookmark")
    #expect(FileManager.default.fileExists(atPath: bookmarkFileURL.path))
    var relaunchedController: SemanticGalleryController? = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: RecordingEmbeddingService(imageVectorsByBytes: [:], textVectors: [:])
        )
    }
    try await waitForCondition("remembered folder to become ready again") {
        await MainActor.run { relaunchedController?.statusStore.status == .ready }
    }
    let relaunchedStatus = await MainActor.run { relaunchedController?.statusStore.status }
    let relaunchedFolder = await MainActor.run { relaunchedController?.libraryStateStore.selectedFolder }
    let relaunchedImageCount = await MainActor.run { relaunchedController?.libraryStateStore.supportedImageCount }
    #expect(relaunchedStatus == .ready)
    #expect(relaunchedFolder?.standardizedFileURL == folder.standardizedFileURL)
    #expect(relaunchedImageCount == 1)
    await MainActor.run {
        firstController = nil
        relaunchedController = nil
    }
    try FileManager.default.removeItem(at: bookmarkFileURL)
    let resetController = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: RecordingEmbeddingService(imageVectorsByBytes: [:], textVectors: [:])
        )
    }
    let resetStatus = await MainActor.run { resetController.statusStore.status }
    let resetFolder = await MainActor.run { resetController.libraryStateStore.selectedFolder }
    #expect(resetStatus == .readyWithoutFolder)
    #expect(resetFolder == nil)
}
@Test
func controllerCreatesSupportRootWhenSelectingFolderFromAnEmptyRuntimeState() async throws {
    let parentRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: parentRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: parentRoot) }
    let supportRoot = parentRoot.appending(path: "Application Support").appending(path: "SemanticGallery")
    let paths = AppPaths(
        supportRoot: supportRoot,
        cachesRoot: supportRoot.appending(path: "Caches"),
        logsRoot: supportRoot.appending(path: "Logs"),
        bundledArtifactsRoot: nil
    )
    let folder = parentRoot.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: ["front": [1.0, 0.0]],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    #expect(FileManager.default.fileExists(atPath: supportRoot.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: paths.cachesRoot.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: paths.logsRoot.path(percentEncoded: false)))
    await controller.selectFolder(at: folder)
    #expect(FileManager.default.fileExists(atPath: paths.databaseURL.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: supportRoot.appending(path: "selected-folder.bookmark").path(percentEncoded: false)))
    #expect(await MainActor.run { controller.workspaceStateStore.results.map(\.filename) } == ["front.jpg"])
}
@Test
func controllerRecreatesAZeroByteDatabaseBeforeSelectingAFolder() async throws {
    let parentRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let root = parentRoot.appending(path: "MissingSupportRoot")
    let supportRoot = root.appending(path: "SemanticGallery")
    let folder = parentRoot.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try FileManager.default.createDirectory(at: supportRoot, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: supportRoot.appending(path: "library.sqlite").path(percentEncoded: false), contents: Data())
    defer { try? FileManager.default.removeItem(at: parentRoot) }
    let paths = AppPaths(root: root)
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: ["front": [1.0, 0.0]],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let database = try LibraryDatabase.open(at: paths.databaseURL)
    try database.migrate()
    let tableNames = try database.tableNames()
    #expect(await MainActor.run { controller.workspaceStateStore.results.map(\.filename) } == ["front.jpg"])
    #expect(tableNames.contains("folders"))
    #expect(FileManager.default.fileExists(atPath: paths.databaseURL.path(percentEncoded: false)))
}
@Test
func controllerKeepsTheChosenFolderWhenEmbeddingPreparationFailsAfterScanning() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: ExplodingEmbeddingService()
        )
    }
    await controller.selectFolder(at: folder)
    let bookmarkURL = paths.supportRoot.appending(path: "selected-folder.bookmark")
    let logURL = paths.logsRoot.appending(path: "semanticgallery.log")
    let status = await MainActor.run { controller.statusStore.status }
    let selectedFolder = await MainActor.run { controller.libraryStateStore.selectedFolder }
    let resultNames = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    let supportedImageCount = await MainActor.run { controller.libraryStateStore.supportedImageCount }
    let searchableImageCount = await MainActor.run { controller.libraryStateStore.searchableImageCount }
    let searchIssue = await MainActor.run { controller.libraryStateStore.searchIssue }
    #expect(status == .ready)
    #expect(selectedFolder == folder)
    #expect(resultNames == ["front.jpg"])
    #expect(supportedImageCount == 1)
    #expect(searchableImageCount == 0)
    #expect(searchIssue == "Semantic search model files could not be loaded from the app bundle.")
    #expect(FileManager.default.fileExists(atPath: bookmarkURL.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: logURL.path(percentEncoded: false)))
    let logText = try String(contentsOf: logURL, encoding: .utf8)
    #expect(logText.contains("Semantic search model files could not be loaded from the app bundle."))
    #expect(logText.contains(folder.path(percentEncoded: false)))
}
@Test
func controllerRebuildsTheRememberedFolderWhenTheDatabaseIsMissingOnRelaunch() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let firstService = RecordingEmbeddingService(
        imageVectorsByBytes: ["front": [1.0, 0.0]],
        textVectors: [:]
    )
    var controller: SemanticGalleryController? = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: firstService
        )
    }
    await controller?.selectFolder(at: folder)
    try FileManager.default.removeItem(at: paths.databaseURL)
    await MainActor.run {
        controller = nil
    }
    let relaunchedService = RecordingEmbeddingService(
        imageVectorsByBytes: ["front": [1.0, 0.0]],
        textVectors: [:]
    )
    let relaunchedController = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: relaunchedService
        )
    }
    try await waitForCondition("remembered folder rebuild to restore results") {
        await MainActor.run { relaunchedController.workspaceStateStore.results.map(\.filename) == ["front.jpg"] }
    }
    #expect(await MainActor.run { relaunchedController.statusStore.status } == .ready)
    #expect(await relaunchedService.imageEncodeCount() > 0)
}
@Test
func controllerDoesNotLetRememberedFolderBootstrapClearANewerFolderSelection() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let rememberedFolder = root.appending(path: "Remembered")
    let newFolder = root.appending(path: "New")
    try FileManager.default.createDirectory(at: rememberedFolder, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: newFolder, withIntermediateDirectories: true)
    try Data("remembered".utf8).write(to: rememberedFolder.appending(path: "remembered.jpg"))
    try Data("new".utf8).write(to: newFolder.appending(path: "new.jpg"))
    let bookmarkStore = FolderBookmarkStore(
        bookmarkFileURL: paths.supportRoot.appending(path: "selected-folder.bookmark")
    )
    try bookmarkStore.saveBookmark(for: rememberedFolder)
    let embeddingService = DelayedOutcomeEmbeddingService(
        imageVectorsByBytes: [
            "new": [1.0, 0.0],
        ],
        failuresByBytes: [
            "remembered": .invalidModelArtifact,
        ],
        delaysByBytes: [
            "remembered": 400_000_000,
        ]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: newFolder)
    try await waitForCondition("new folder selection stays active after bootstrap finishes") {
        await MainActor.run {
            controller.statusStore.status == .ready
                && controller.libraryStateStore.selectedFolder?.standardizedFileURL == newFolder.standardizedFileURL
                && controller.workspaceStateStore.results.map(\.filename) == ["new.jpg"]
        }
    }
    let status = await MainActor.run { controller.statusStore.status }
    let selectedFolder = await MainActor.run { controller.libraryStateStore.selectedFolder }
    let resultNames = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    #expect(status == .ready)
    #expect(selectedFolder?.standardizedFileURL == newFolder.standardizedFileURL)
    #expect(resultNames == ["new.jpg"])
}
@Test
func controllerSelectingTheSameFolderAgainReindexesNewFiles() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "new": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0,
                folderObservationDebounceNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    try Data("new".utf8).write(to: folder.appending(path: "new.jpg"))
    await controller.selectFolder(at: folder)
    let filenames = await MainActor.run { controller.workspaceStateStore.results.map(\.filename).sorted() }
    let supportedImageCount = await MainActor.run { controller.libraryStateStore.supportedImageCount }
    #expect(filenames == ["front.jpg", "new.jpg"])
    #expect(supportedImageCount == 2)
}
@Test
func controllerObservedFolderChangesRefreshVisibleResults() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let folderChangeMonitor = TestFolderChangeMonitor()
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0,
                folderObservationDebounceNanoseconds: 0
            ),
            embeddingService: embeddingService,
            folderChangeMonitor: folderChangeMonitor
        )
    }
    await controller.selectFolder(at: folder)
    try Data("back".utf8).write(to: folder.appending(path: "back.jpg"))
    folderChangeMonitor.emit(changeCount: 1)
    try await waitForCondition("observed folder changes to add a visible result") {
        await MainActor.run {
            controller.workspaceStateStore.results.map(\.filename).sorted() == ["back.jpg", "front.jpg"]
                && controller.libraryStateStore.supportedImageCount == 2
        }
    }
    try FileManager.default.removeItem(at: folder.appending(path: "front.jpg"))
    folderChangeMonitor.emit(changeCount: 1)
    try await waitForCondition("observed folder changes to remove a visible result") {
        await MainActor.run {
            controller.workspaceStateStore.results.map(\.filename) == ["back.jpg"]
                && controller.libraryStateStore.supportedImageCount == 1
        }
    }
}
@Test
func controllerSearchingSimilarFromPreviewUsesTheCurrentImage() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let previewItem = try #require(
        await MainActor.run { controller.workspaceStateStore.results.first }
    )
    await MainActor.run {
        controller.openPreview(assetID: previewItem.assetID)
        controller.searchSimilarToPreviewItem()
    }
    try await waitForCondition("preview similar search results") {
        await MainActor.run {
            controller.workspaceStateStore.previewAssetID == nil
                && controller.workspaceStateStore.results.first?.filename == previewItem.filename
                && controller.workspaceStateStore.pastedImageData != nil
        }
    }
    #expect(await embeddingService.pastedImageEncodeCount() == 1)
}
@Test
func controllerDeletingPreviewMovesTheCurrentImageToTrash() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let previewItem = try #require(
        await MainActor.run { controller.workspaceStateStore.results.first }
    )
    await MainActor.run {
        controller.openPreview(assetID: previewItem.assetID)
        controller.deletePreviewItem()
    }
    try await waitForCondition("preview delete refreshes results") {
        await MainActor.run {
            controller.workspaceStateStore.previewAssetID == nil
                && controller.workspaceStateStore.results.count == 1
        }
    }
    let database = try LibraryDatabase.open(at: paths.databaseURL)
    try database.migrate()
    let visible = try database.visibleFileInstances(
        inFolderAbsolutePath: folder.path(percentEncoded: false),
        limit: 10
    )
    #expect(FileManager.default.fileExists(atPath: previewItem.absolutePath) == false)
    #expect(visible.count == 1)
}
@Test
func controllerReusesStoredEmbeddingsAndRunsImageVectorSearch() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("shared".utf8).write(to: folder.appending(path: "one.jpg"))
    try Data("shared".utf8).write(to: folder.appending(path: "duplicate.jpg"))
    try Data("green".utf8).write(to: folder.appending(path: "two.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "shared": [1.0, 0.0],
            "green": [0.0, 1.0],
            "query-image": [1.0, 0.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 150_000_000
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let firstImageEncodeCount = await embeddingService.imageEncodeCount()
    #expect(firstImageEncodeCount == 2)
    #expect(
        await MainActor.run { controller.workspaceStateStore.results.map(\.filename) } == [
            "duplicate.jpg",
            "one.jpg",
            "two.png",
        ]
    )
    await controller.selectFolder(at: folder)
    let secondImageEncodeCount = await embeddingService.imageEncodeCount()
    #expect(secondImageEncodeCount == 2)
    await MainActor.run {
        controller.workspaceStateStore.queryText = ""
        controller.workspaceStateStore.pastedImageData = Data("query-image".utf8)
        controller.runSearch()
    }
    try await waitForCondition("semantic image search results") {
        await MainActor.run { controller.workspaceStateStore.results.first?.filename == "duplicate.jpg" }
    }
    let semanticResults = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    #expect(semanticResults == ["duplicate.jpg", "one.jpg", "two.png"])
}
@Test
func controllerFolderPreparationReportsLiveProgressFromRealCounts() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("amber".utf8).write(to: folder.appending(path: "amber.jpg"))
    try Data("jade".utf8).write(to: folder.appending(path: "jade.png"))
    let database = try LibraryDatabase.open(at: paths.databaseURL)
    try database.migrate()
    _ = try await FolderIndexer(database: database).rebuildIndex(for: folder)
    let visible = try database.visibleFileInstances(inFolderAbsolutePath: folder.path(percentEncoded: false), limit: 10)
    let reusedAssetID = try #require(visible.first?.assetID)
    try database.upsertEmbedding(assetID: reusedAssetID, encoderVersion: "stage1", vector: [1.0, 0.0])
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "amber": [1.0, 0.0],
            "jade": [0.0, 1.0],
        ],
        textVectors: [:],
        imageDelayNanoseconds: 250_000_000
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    let selectTask = Task {
        await controller.selectFolder(at: folder)
    }
    try await waitForCondition("folder progress shows scan and reuse counts", timeoutNanoseconds: 5_000_000_000) {
        await MainActor.run {
            let progress = controller.libraryStateStore.folderPreparationProgress
            guard
                let scan = progress.first(where: { $0.step == .scanSupportedImages }),
                let reuse = progress.first(where: { $0.step == .reuseExistingEmbeddings })
            else {
                return false
            }
            return scan.message == "Scanned 2 of 2 images · 2 unique assets"
                && reuse.message == "Reused 1 embeddings, 1 still needed"
                && scan.elapsedSeconds != nil
        }
    }
    await selectTask.value
    let status = await MainActor.run { controller.statusStore.status }
    let results = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    #expect(status == .ready)
    #expect(results == ["amber.jpg", "jade.png"])
    #expect(await embeddingService.imageEncodeCount() == 1)
}
@Test
func controllerShowsChosenFolderWhilePreparationIsStillRunning() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("slow".utf8).write(to: folder.appending(path: "slow.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: ["slow": [1.0, 0.0]],
        textVectors: [:],
        imageDelayNanoseconds: 1_200_000_000
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    let selectTask = Task {
        await controller.selectFolder(at: folder)
    }
    try await waitForCondition("selected folder becomes visible during preparation", timeoutNanoseconds: 2_000_000_000) {
        await MainActor.run {
            controller.libraryStateStore.selectedFolder == folder
                && controller.libraryStateStore.folderPreparationProgress.isEmpty == false
                && controller.statusStore.status == .indexing
        }
    }
    await selectTask.value
    let finalResults = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    #expect(finalResults == ["slow.jpg"])
}
@Test
func controllerCompletesFastFolderPreparationWithoutArtificialDelay() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "FastLibrary")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [:],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    let selectionStart = Date()
    await controller.selectFolder(at: folder)
    let elapsed = Date().timeIntervalSince(selectionStart)
    #expect(elapsed < 1.0)
    #expect(await MainActor.run { controller.statusStore.status } == .ready)
    #expect(await MainActor.run { controller.libraryStateStore.selectedFolder } == folder)
    #expect(await MainActor.run { controller.libraryStateStore.folderPreparationProgress.isEmpty })
}
@Test
func controllerClearsCompletedFolderPreparationOnceIndexingFinishes() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let status = await MainActor.run { controller.statusStore.status }
    let folderPreparationProgress = await MainActor.run { controller.libraryStateStore.folderPreparationProgress }
    #expect(status == .ready)
    #expect(folderPreparationProgress.isEmpty)
}
@Test
func controllerPublishesFolderPreparationProgressWhileIndexingAndClearsItAtCompletion() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("slow".utf8).write(to: folder.appending(path: "slow.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: ["slow": [1.0, 0.0]],
        textVectors: [:],
        imageDelayNanoseconds: 1_200_000_000
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    let selectTask = Task {
        await controller.selectFolder(at: folder)
    }
    try await waitForCondition("folder progress visible while indexing", timeoutNanoseconds: 2_000_000_000) {
        await MainActor.run {
            controller.statusStore.status == .indexing
                && controller.libraryStateStore.folderPreparationProgress.isEmpty == false
        }
    }
    await selectTask.value
    let status = await MainActor.run { controller.statusStore.status }
    let folderPreparationProgress = await MainActor.run { controller.libraryStateStore.folderPreparationProgress }
    #expect(status == .ready)
    #expect(folderPreparationProgress.isEmpty)
}
@Test
func controllerKeepsTheFolderSelectedWhenSomeImagesCannotBeEmbedded() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("broken".utf8).write(to: folder.appending(path: "broken.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
        ],
        textVectors: [:],
        invalidImageBytes: ["broken"]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let status = await MainActor.run { controller.statusStore.status }
    let selectedFolder = await MainActor.run { controller.libraryStateStore.selectedFolder }
    let results = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    let supportedImageCount = await MainActor.run { controller.libraryStateStore.supportedImageCount }
    let searchableImageCount = await MainActor.run { controller.libraryStateStore.searchableImageCount }
    let searchIssue = await MainActor.run { controller.libraryStateStore.searchIssue }
    #expect(status == .ready)
    #expect(selectedFolder == folder)
    #expect(results == ["broken.jpg", "front.jpg"])
    #expect(supportedImageCount == 2)
    #expect(searchableImageCount == 1)
    #expect(searchIssue == nil)
}
@Test
func controllerPrewarmsTheEmbeddingServiceAfterFolderPreparation() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("existing".utf8).write(to: folder.appending(path: "existing.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "existing": [1.0, 0.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    try await waitForCondition("embedding service prewarm") {
        await embeddingService.prepareCallCount() > 0
    }
}
@Test
func controllerTextSearchKeepsExactFilenameMatchesAheadOfSemanticFallback() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("alpha".utf8).write(to: folder.appending(path: "sample-1.jpg"))
    try Data("beta".utf8).write(to: folder.appending(path: "sample-2.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        ],
        textVectors: [
            "sample-1": [0.0, 1.0],
        ]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    await MainActor.run {
        controller.workspaceStateStore.queryText = "sample-1"
        controller.runSearch()
    }
    try await waitForCondition("filename search results") {
        await MainActor.run { controller.workspaceStateStore.results.isEmpty == false }
    }
    let results = await MainActor.run { controller.workspaceStateStore.results.map(\.filename) }
    #expect(results == ["sample-1.jpg"])
}
@Test
func controllerCachesVectorQueriesAcrossRepeatedSearches() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("forest".utf8).write(to: folder.appending(path: "forest.jpg"))
    try Data("river".utf8).write(to: folder.appending(path: "river.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "forest": [1.0, 0.0],
            "river": [0.0, 1.0],
            "query-image": [0.0, 1.0],
        ],
        textVectors: [
            "warm evening": [1.0, 0.0],
        ]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    await MainActor.run {
        controller.workspaceStateStore.queryText = "warm evening"
        controller.runSearch()
    }
    try await waitForCondition("first semantic text search results") {
        await MainActor.run { controller.workspaceStateStore.results.first?.filename == "forest.jpg" }
    }
    await MainActor.run {
        controller.runSearch()
    }
    try await waitForCondition("second semantic text search results") {
        await MainActor.run { controller.workspaceStateStore.results.first?.filename == "forest.jpg" }
    }
    #expect(await embeddingService.textEncodeCount() == 1)
    await MainActor.run {
        controller.workspaceStateStore.queryText = ""
        controller.workspaceStateStore.pastedImageData = Data("query-image".utf8)
        controller.runSearch()
    }
    try await waitForCondition("first image semantic search results") {
        await MainActor.run { controller.workspaceStateStore.results.first?.filename == "river.jpg" }
    }
    await MainActor.run {
        controller.runSearch()
    }
    try await waitForCondition("second image semantic search results") {
        await MainActor.run { controller.workspaceStateStore.results.first?.filename == "river.jpg" }
    }
    #expect(await embeddingService.pastedImageEncodeCount() == 1)
}
@Test
func controllerLogsWhenSemanticSearchHasNoEmbeddingsAvailable() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [
            "warm evening": [1.0, 0.0],
        ],
        invalidImageBytes: ["front", "back"]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    await MainActor.run {
        controller.workspaceStateStore.queryText = "warm evening"
        controller.runSearch()
    }
    try await waitForCondition("semantic search without embeddings") {
        await MainActor.run { controller.workspaceStateStore.isSearching == false }
    }
    let logText = try String(
        contentsOf: paths.logsRoot.appending(path: "semanticgallery.log"),
        encoding: .utf8
    )
    #expect(logText.contains("Semantic search could not run because no searchable embeddings are available."))
    #expect(logText.contains(folder.path(percentEncoded: false)))
}
@Test
func controllerLogsFolderPreparationProgressDuringEmbeddingGeneration() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let logText = try String(
        contentsOf: paths.logsRoot.appending(path: "semanticgallery.log"),
        encoding: .utf8
    )
    #expect(logText.contains("Preparing the selected folder for search."))
    #expect(logText.contains("Generating search embeddings"))
    #expect(logText.contains("Prepared 1 of 1 embeddings"))
    #expect(logText.contains("step=generate_embeddings"))
}
@Test
func controllerBlocksPrivateAdaptationWhenTheFolderHasFewerThanOneHundredImages() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "SmallLibrary")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    for index in 0..<99 {
        try Data("small-\(index)".utf8).write(to: folder.appending(path: "small-\(index).jpg"))
    }
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [:],
        textVectors: [:]
    )
    let adaptationTrainer = RecordingAdaptationTrainer()
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService,
            adaptationTrainer: adaptationTrainer
        )
    }
    await controller.selectFolder(at: folder)
    await MainActor.run {
        controller.startPrivateAdaptation()
    }
    let notice = await MainActor.run { controller.libraryStateStore.privateAdaptationNotice }
    let status = await MainActor.run { controller.statusStore.status }
    #expect(notice?.contains("99 supported images") == true)
    #expect(status == .ready)
    #expect(await adaptationTrainer.runCount() == 0)
}
@Test
func controllerUninstallClearsInstalledStateAndReturnsToInstallRequired() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("sample".utf8).write(to: folder.appending(path: "sample.jpg"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: RecordingEmbeddingService(imageVectorsByBytes: [:], textVectors: [:])
        )
    }
    await controller.selectFolder(at: folder)
    await MainActor.run {
        controller.startUninstall()
    }
    let clearedState = await MainActor.run {
        (
            status: controller.statusStore.status,
            selectedFolder: controller.libraryStateStore.selectedFolder,
            supportedImageCount: controller.libraryStateStore.supportedImageCount,
            searchableImageCount: controller.libraryStateStore.searchableImageCount,
            activeEncoderVersion: controller.libraryStateStore.activeEncoderVersion,
            searchIssue: controller.libraryStateStore.searchIssue,
            resultsCount: controller.workspaceStateStore.results.count,
            selectionCount: controller.workspaceStateStore.selectedAssetIDs.count,
            pastedImageData: controller.workspaceStateStore.pastedImageData
        )
    }
    #expect(clearedState.status == .readyWithoutFolder)
    #expect(clearedState.selectedFolder == nil)
    #expect(clearedState.supportedImageCount == nil)
    #expect(clearedState.searchableImageCount == nil)
    #expect(clearedState.activeEncoderVersion == nil)
    #expect(clearedState.searchIssue == nil)
    #expect(clearedState.resultsCount == 0)
    #expect(clearedState.selectionCount == 0)
    #expect(clearedState.pastedImageData == nil)
    #expect(FileManager.default.fileExists(atPath: paths.supportRoot.path) == false)
    #expect(FileManager.default.fileExists(atPath: paths.installStateURL.path) == false)
    let relaunchedController = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: RecordingEmbeddingService(imageVectorsByBytes: [:], textVectors: [:])
        )
    }
    let relaunchedStatus = await MainActor.run { relaunchedController.statusStore.status }
    #expect(relaunchedStatus == .readyWithoutFolder)
}
@Test
func controllerRunsPrivateAdaptationAndSwitchesToTheFolderSpecificEncoder() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "PrivateAlbum")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    for index in 0..<120 {
        try Data("private-\(index)".utf8).write(to: folder.appending(path: "private-\(index).jpg"))
    }
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let stage1EmbeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [:],
        textVectors: [:]
    )
    let stage2EmbeddingService = RecordingEmbeddingService(
        encoderVersion: "stage2-privatealbum-abcdef",
        imageVectorsByBytes: [:],
        textVectors: [:]
    )
    let adaptationTrainer = RecordingAdaptationTrainer(
        result: .init(
            artifact: FolderAdaptationArtifact(
                folderKey: "privatealbum-1234567890ab",
                folderPath: folder.path(percentEncoded: false),
                encoderVersion: "stage2-privatealbum-abcdef",
                adapterWeightsURL: paths.modelsRoot
                    .appending(path: "Adapted")
                    .appending(path: "privatealbum-1234567890ab")
                    .appending(path: "weights.safetensors"),
                summaryURL: paths.modelsRoot
                    .appending(path: "Adapted")
                    .appending(path: "privatealbum-1234567890ab")
                    .appending(path: "summary.json"),
                trainedImageCount: 100
            ),
            embeddingService: stage2EmbeddingService
        )
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: stage1EmbeddingService,
            adaptationTrainer: adaptationTrainer
        )
    }
    await controller.selectFolder(at: folder)
    await MainActor.run {
        controller.startPrivateAdaptation()
    }
    try await waitForCondition("private adaptation finishes") {
        await MainActor.run {
            controller.libraryStateStore.activeEncoderVersion == "stage2-privatealbum-abcdef"
                && controller.statusStore.status == .ready
        }
    }
    let status = await MainActor.run { controller.statusStore.status }
    let activeEncoderVersion = await MainActor.run { controller.libraryStateStore.activeEncoderVersion }
    #expect(status == .ready)
    #expect(activeEncoderVersion == "stage2-privatealbum-abcdef")
    #expect(await adaptationTrainer.runCount() == 1)
}
@Test
func controllerEnteringSelectionModeClearsPreviewAndMetadata() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let firstAssetID = try #require(
        await MainActor.run { controller.workspaceStateStore.results.first?.assetID }
    )
    await MainActor.run {
        controller.openPreview(assetID: firstAssetID)
        controller.togglePreviewMetadata()
        controller.enterSelectionMode()
    }
    let previewAssetID = await MainActor.run { controller.workspaceStateStore.previewAssetID }
    let isPreviewMetadataVisible = await MainActor.run { controller.workspaceStateStore.isPreviewMetadataVisible }
    let isSelectionModeEnabled = await MainActor.run { controller.workspaceStateStore.isSelectionModeEnabled }
    let selectionCount = await MainActor.run { controller.workspaceStateStore.selectionCount }
    #expect(previewAssetID == nil)
    #expect(isPreviewMetadataVisible == false)
    #expect(isSelectionModeEnabled == true)
    #expect(selectionCount == 0)
}
@Test
func controllerLeavingSelectionModeClearsSelectionAndOpeningPreviewLeavesBrowseState() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let paths = AppPaths(root: root)
    let coordinator = InstallCoordinator(paths: paths, downloader: ArtifactDownloader.stubbed)
    _ = try await coordinator.prepare()
    let folder = root.appending(path: "Library")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data("front".utf8).write(to: folder.appending(path: "front.jpg"))
    try Data("back".utf8).write(to: folder.appending(path: "back.png"))
    let suiteName = "SemanticGalleryControllerTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let embeddingService = RecordingEmbeddingService(
        imageVectorsByBytes: [
            "front": [1.0, 0.0],
            "back": [0.0, 1.0],
        ],
        textVectors: [:]
    )
    let controller = await MainActor.run {
        SemanticGalleryController(
            runtimeOptions: .init(
                paths: paths,
                defaultsSuiteName: suiteName,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: embeddingService
        )
    }
    await controller.selectFolder(at: folder)
    let firstAssetID = try #require(
        await MainActor.run { controller.workspaceStateStore.results.first?.assetID }
    )
    await MainActor.run {
        controller.enterSelectionMode()
        controller.toggleSelection(assetID: firstAssetID)
    }
    let selectedCountBeforeLeaving = await MainActor.run { controller.workspaceStateStore.selectionCount }
    #expect(selectedCountBeforeLeaving == 1)
    await MainActor.run {
        controller.leaveSelectionMode()
        controller.openPreview(assetID: firstAssetID)
    }
    let selectedCountAfterLeaving = await MainActor.run { controller.workspaceStateStore.selectionCount }
    let isSelectionModeEnabled = await MainActor.run { controller.workspaceStateStore.isSelectionModeEnabled }
    let previewAssetID = await MainActor.run { controller.workspaceStateStore.previewAssetID }
    #expect(selectedCountAfterLeaving == 0)
    #expect(isSelectionModeEnabled == false)
    #expect(previewAssetID == firstAssetID)
}

}

private func waitForCondition(
    _ description: String,
    timeoutNanoseconds: UInt64 = 2_000_000_000,
    condition: @escaping @Sendable () async -> Bool
) async throws {
    let start = DispatchTime.now().uptimeNanoseconds
    while await condition() == false {
        try await Task.sleep(nanoseconds: 50_000_000)
        if DispatchTime.now().uptimeNanoseconds - start >= timeoutNanoseconds {
            throw ConditionTimeout(description: description)
        }
    }
}
private struct ConditionTimeout: Error, CustomStringConvertible {
    let description: String
}
private actor RecordingEmbeddingService: GalleryEmbeddingService {
    nonisolated let encoderVersion: String
    private let imageVectorsByBytes: [String: [Double]]
    private let textVectors: [String: [Double]]
    private let imageDelayNanoseconds: UInt64
    private let invalidImageBytes: Set<String>
    private var imageRequests: [String] = []
    private var pastedImageRequests: [String] = []
    private var textRequests: [String] = []
    private var prepareCalls = 0
    init(
        encoderVersion: String = "stage1",
        imageVectorsByBytes: [String: [Double]],
        textVectors: [String: [Double]],
        imageDelayNanoseconds: UInt64 = 0,
        invalidImageBytes: Set<String> = []
    ) {
        self.encoderVersion = encoderVersion
        self.imageVectorsByBytes = imageVectorsByBytes
        self.textVectors = textVectors
        self.imageDelayNanoseconds = imageDelayNanoseconds
        self.invalidImageBytes = invalidImageBytes
    }
    func encodeImage(at url: URL) async throws -> [Double] {
        let key = try String(decoding: Data(contentsOf: url), as: UTF8.self)
        if invalidImageBytes.contains(key) {
            throw GalleryEmbeddingError.invalidImageData
        }
        if imageDelayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: imageDelayNanoseconds)
        }
        imageRequests.append(key)
        return imageVectorsByBytes[key, default: [0.0, 0.0]]
    }
    func encodeImage(data: Data) async throws -> [Double] {
        let key = String(decoding: data, as: UTF8.self)
        pastedImageRequests.append(key)
        return imageVectorsByBytes[key, default: [0.0, 0.0]]
    }
    func encodeText(_ text: String) async throws -> [Double] {
        textRequests.append(text)
        return textVectors[text, default: [0.0, 0.0]]
    }
    func prepareForQueries() async throws {
        prepareCalls += 1
    }
    func imageEncodeCount() -> Int {
        imageRequests.count
    }
    func pastedImageEncodeCount() -> Int {
        pastedImageRequests.count
    }
    func textEncodeCount() -> Int {
        textRequests.count
    }
    func prepareCallCount() -> Int {
        prepareCalls
    }
}
private actor ExplodingEmbeddingService: GalleryEmbeddingService {
    nonisolated let encoderVersion = "stage1"
    func encodeImage(at url: URL) async throws -> [Double] {
        throw GalleryEmbeddingError.invalidModelArtifact
    }
    func encodeImage(data: Data) async throws -> [Double] {
        throw GalleryEmbeddingError.invalidModelArtifact
    }
    func encodeText(_ text: String) async throws -> [Double] {
        [0.0, 0.0]
    }
    func prepareForQueries() async throws {}
}
private actor DelayedOutcomeEmbeddingService: GalleryEmbeddingService {
    nonisolated let encoderVersion = "stage1"
    private let imageVectorsByBytes: [String: [Double]]
    private let failuresByBytes: [String: GalleryEmbeddingError]
    private let delaysByBytes: [String: UInt64]
    init(
        imageVectorsByBytes: [String: [Double]],
        failuresByBytes: [String: GalleryEmbeddingError],
        delaysByBytes: [String: UInt64] = [:]
    ) {
        self.imageVectorsByBytes = imageVectorsByBytes
        self.failuresByBytes = failuresByBytes
        self.delaysByBytes = delaysByBytes
    }
    func encodeImage(at url: URL) async throws -> [Double] {
        let key = try String(decoding: Data(contentsOf: url), as: UTF8.self)
        if let delay = delaysByBytes[key], delay > 0 {
            try await Task.sleep(nanoseconds: delay)
        }
        if let failure = failuresByBytes[key] {
            throw failure
        }
        return imageVectorsByBytes[key, default: [0.0, 0.0]]
    }
    func encodeImage(data: Data) async throws -> [Double] {
        let key = String(decoding: data, as: UTF8.self)
        if let failure = failuresByBytes[key] {
            throw failure
        }
        return imageVectorsByBytes[key, default: [0.0, 0.0]]
    }
    func encodeText(_ text: String) async throws -> [Double] {
        [0.0, 0.0]
    }
    func prepareForQueries() async throws {}
}
private final class TestFolderChangeMonitor: FolderChangeMonitoring, @unchecked Sendable {
    private var onChange: (@Sendable (Int) -> Void)?
    func startMonitoring(folderURL: URL, onChange: @escaping @Sendable (Int) -> Void) throws {
        self.onChange = onChange
    }
    func stopMonitoring() {
        onChange = nil
    }
    func emit(changeCount: Int) {
        onChange?(changeCount)
    }
}
private actor RecordingAdaptationTrainer: GalleryAdaptationTraining {
    private let result: GalleryAdaptationResult?
    private(set) var calls: [URL] = []
    init(result: GalleryAdaptationResult? = nil) {
        self.result = result
    }
    func runAdaptation(
        for folderURL: URL,
        progress: @escaping @Sendable (PrivateAdaptationProgress) async -> Void
    ) async throws -> GalleryAdaptationResult {
        calls.append(folderURL)
        await progress(
            PrivateAdaptationProgress(
                step: .prepareData,
                message: "Preparing adaptation data",
                stepProgress: 1,
                overallProgress: 0.34,
                elapsedSeconds: 1,
                remainingSeconds: 2
            )
        )
        let result = try #require(result)
        return result
    }
    func existingArtifact(for folderURL: URL) async throws -> GalleryAdaptationResult? {
        nil
    }
    func runCount() -> Int {
        calls.count
    }
}
