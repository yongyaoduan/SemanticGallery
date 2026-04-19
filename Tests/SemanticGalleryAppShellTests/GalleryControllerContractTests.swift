import Foundation
import Testing
@testable import SemanticGalleryAppShell
@testable import SemanticGalleryML
@testable import SemanticGalleryPersistence
@testable import SemanticGallerySearch

@Suite("Gallery Controller Contracts", .serialized)
struct GalleryControllerContractTests {
    /// Formal specification for callers:
    /// Pre: `selectedFolder = nil`, `results = R`, `selectedResultIDs = S`, and `previewResultID = p`.
    /// Post after `runSearch()`:
    /// `results' = [] ∧ selectedResultIDs' = ∅ ∧ queryText' = queryText`.
    @Test
    func runSearchWithoutASelectedFolderClearsVisibleResultsAndSelectionOnly() async throws {
        let controller = await makeController()
        await MainActor.run {
            controller.usageState.results = [
                SearchAsset(assetID: 1, absolutePath: "/tmp/library/one.jpg", thumbnailPath: nil)
            ]
            controller.usageState.selectedResultIDs = [1]
            controller.usageState.previewResultID = 1
            controller.usageState.queryText = "cat"
        }

        await MainActor.run {
            controller.runSearch()
        }

        let results = await MainActor.run { controller.usageState.results }
        let selected = await MainActor.run { controller.usageState.selectedResultIDs }
        let query = await MainActor.run { controller.usageState.queryText }
        #expect(results.isEmpty)
        #expect(selected.isEmpty)
        #expect(query == "cat")
    }

    /// Formal specification for callers:
    /// Pre: `selectedFolder = nil`.
    /// Post after `startPrivateAdaptation()`:
    /// `privateAdaptationNotice' ≠ nil ∧ privateAdaptationProgress' = [] ∧ status' = readyWithoutFolder`.
    @Test
    func startPrivateAdaptationWithoutAFolderLeavesTheControllerInTheEmptyLibraryState() async throws {
        let controller = await makeController()

        await MainActor.run {
            controller.startPrivateAdaptation()
        }

        let notice = await MainActor.run { controller.libraryState.privateAdaptationNotice }
        let progress = await MainActor.run { controller.libraryState.privateAdaptationProgress }
        let status = await MainActor.run { controller.statusState.status }
        #expect(notice == "Choose a folder in Settings before starting private adaptation.")
        #expect(progress.isEmpty)
        #expect(status == .readyWithoutFolder)
    }

    /// Formal specification for callers:
    /// Pre: `selectedFolder = f` and `supportedImageCount = n < 100`.
    /// Post after `startPrivateAdaptation()`:
    /// `status' = ready ∧ privateAdaptationProgress' = [] ∧ privateAdaptationNotice'` explains `n`.
    @Test
    func startPrivateAdaptationWithTooFewImagesProducesOnlyANotice() async throws {
        let controller = await makeController()
        let folder = URL(fileURLWithPath: "/tmp/library", isDirectory: true)

        await MainActor.run {
            controller.libraryState.selectedFolder = folder
            controller.libraryState.supportedImageCount = 12
            controller.statusState.status = .ready
            controller.startPrivateAdaptation()
        }

        let notice = await MainActor.run { controller.libraryState.privateAdaptationNotice }
        let progress = await MainActor.run { controller.libraryState.privateAdaptationProgress }
        let status = await MainActor.run { controller.statusState.status }
        #expect(notice == "This folder currently has 12 supported images. SemanticGallery needs at least 100 images before local adaptation can begin.")
        #expect(progress.isEmpty)
        #expect(status == .ready)
    }

    /// Formal specification for callers:
    /// Pre: `selectedResultIDs = ∅`.
    /// Post after `deleteSelectedResults()`:
    /// `results' = results ∧ previewResultID' = previewResultID`.
    @Test
    func deleteSelectedResultsWithAnEmptySelectionIsANoOp() async throws {
        let controller = await makeController()
        await MainActor.run {
            controller.usageState.results = [
                SearchAsset(assetID: 1, absolutePath: "/tmp/library/one.jpg", thumbnailPath: nil)
            ]
            controller.usageState.previewResultID = 1
        }

        await MainActor.run {
            controller.deleteSelectedResults()
        }

        let results = await MainActor.run { controller.usageState.results }
        let previewID = await MainActor.run { controller.usageState.previewResultID }
        #expect(results.count == 1)
        #expect(previewID == 1)
    }

    @MainActor
    private func makeController() -> GalleryController {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        return GalleryController(
            runtimeOptions: RuntimeConfiguration(
                paths: AppPaths(root: root),
                defaultsSuiteName: nil,
                useStubDownloads: true,
                installStepDelayNanoseconds: 0,
                folderPreparationDelayNanoseconds: 0
            ),
            embeddingService: StubEmbeddingService()
        )
    }
}

private actor StubEmbeddingService: GalleryEmbeddingService {
    nonisolated let encoderVersion = "stage1"

    func encodeImage(at url: URL) async throws -> [Double] {
        [0.0, 0.0]
    }

    func encodeImage(data: Data) async throws -> [Double] {
        [0.0, 0.0]
    }

    func encodeText(_ text: String) async throws -> [Double] {
        [0.0, 0.0]
    }

    func prepareForQueries() async throws {}
}
