import Foundation
import Testing
@testable import SemanticGallerySearch

@MainActor
@Suite("Usage State Contracts")
struct UsageStateContractTests {
    /// Formal specification for callers:
    /// Pre: `results = [r0, r1]` and `selectedResultIDs = ∅`.
    /// Post after `toggleSelection(r0.id)`:
    /// `selectedResultIDs' = { r0.id }` and `selectionCount' = 1`.
    /// Post after a second `toggleSelection(r0.id)`:
    /// `selectedResultIDs'' = ∅` and `selectionCount'' = 0`.
    @Test
    func toggleSelectionIsAnInvolutionOnTheTargetResult() {
        let state = UsageState(results: sampleResults)

        state.toggleSelection(resultID: sampleResults[0].id)
        #expect(state.selectedResultIDs == [sampleResults[0].id])
        #expect(state.selectionCount == 1)
        #expect(state.isSelected(sampleResults[0].id))

        state.toggleSelection(resultID: sampleResults[0].id)
        #expect(state.selectedResultIDs.isEmpty)
        #expect(state.selectionCount == 0)
        #expect(state.isSelected(sampleResults[0].id) == false)
    }

    /// Formal specification for callers:
    /// Pre: `results = R`, `previewResultID ∈ ids(R)`, and `selectedResultIDs` may be non-empty.
    /// Post after `enterSelectionMode()`:
    /// `isSelectionModeEnabled' = true ∧ selectedResultIDs' = ∅ ∧ previewResultID' = nil`.
    /// Post after `leaveSelectionMode()`:
    /// `isSelectionModeEnabled'' = false ∧ selectedResultIDs'' = ∅`.
    @Test
    func selectionModeTransitionsClearPreviewAndSelection() {
        let state = UsageState(
            results: sampleResults,
            selectedResultIDs: [sampleResults[0].id],
            isSelectionModeEnabled: false,
            previewResultID: sampleResults[1].id,
            isPreviewMetadataVisible: true
        )

        state.enterSelectionMode()
        #expect(state.isSelectionModeEnabled)
        #expect(state.selectedResultIDs.isEmpty)
        #expect(state.previewResultID == nil)
        #expect(state.isPreviewMetadataVisible == false)

        state.toggleSelection(resultID: sampleResults[0].id)
        state.leaveSelectionMode()
        #expect(state.isSelectionModeEnabled == false)
        #expect(state.selectedResultIDs.isEmpty)
    }

    /// Formal specification for callers:
    /// Pre: `results = R` and `resultID ∈ ids(R)`.
    /// Post after `openPreview(resultID)`:
    /// `previewResultID' = resultID ∧ previewItem'.id = resultID ∧ selectedResultIDs' = ∅ ∧ isSelectionModeEnabled' = false`.
    /// Post after `closePreview()`:
    /// `previewResultID'' = nil ∧ isPreviewMetadataVisible'' = false`.
    @Test
    func previewLifecyclePinsTheRequestedResultAndResetsTransientUiState() {
        let state = UsageState(
            results: sampleResults,
            selectedResultIDs: [sampleResults[0].id],
            isSearching: false,
            isSelectionModeEnabled: true,
            previewResultID: nil,
            isPreviewMetadataVisible: true
        )

        state.openPreview(resultID: sampleResults[1].id)
        #expect(state.previewResultID == sampleResults[1].id)
        #expect(state.previewItem?.id == sampleResults[1].id)
        #expect(state.selectedResultIDs.isEmpty)
        #expect(state.isSelectionModeEnabled == false)
        #expect(state.isPreviewMetadataVisible == false)

        state.closePreview()
        #expect(state.previewResultID == nil)
        #expect(state.previewItem == nil)
        #expect(state.isPreviewMetadataVisible == false)
    }

    /// Formal specification for callers:
    /// Pre: `previewResultID = nil`.
    /// Post after `togglePreviewMetadata()`:
    /// `isPreviewMetadataVisible' = false`.
    /// Pre': `previewResultID ∈ ids(results)`.
    /// Post' after `togglePreviewMetadata()`:
    /// `isPreviewMetadataVisible'' = ¬isPreviewMetadataVisible'`.
    @Test
    func previewMetadataVisibilityRequiresAnOpenPreview() {
        let state = UsageState(results: sampleResults)

        state.togglePreviewMetadata()
        #expect(state.isPreviewMetadataVisible == false)

        state.openPreview(resultID: sampleResults[0].id)
        state.togglePreviewMetadata()
        #expect(state.isPreviewMetadataVisible)

        state.togglePreviewMetadata()
        #expect(state.isPreviewMetadataVisible == false)
    }

    /// Formal specification for callers:
    /// Pre: `results = [r0, r1]`, `previewResultID = r0.id`, and `isPreviewMetadataVisible = true`.
    /// Post after `showNextPreviewItem()`:
    /// `previewResultID' = r1.id ∧ isPreviewMetadataVisible' = false`.
    /// Pre': `previewResultID' = r1.id`.
    /// Post' after `showPreviousPreviewItem()`:
    /// `previewResultID'' = r0.id ∧ isPreviewMetadataVisible'' = false`.
    @Test
    func previewNavigationMovesWithinTheVisibleResultOrder() {
        let state = UsageState(
            results: sampleResults,
            previewResultID: sampleResults[0].id,
            isPreviewMetadataVisible: true
        )

        state.showNextPreviewItem()
        #expect(state.previewResultID == sampleResults[1].id)
        #expect(state.previewItem?.id == sampleResults[1].id)
        #expect(state.isPreviewMetadataVisible == false)

        state.togglePreviewMetadata()
        state.showPreviousPreviewItem()
        #expect(state.previewResultID == sampleResults[0].id)
        #expect(state.previewItem?.id == sampleResults[0].id)
        #expect(state.isPreviewMetadataVisible == false)
    }

    /// Formal specification for callers:
    /// Pre: `results = R`.
    /// Post after `selectAllVisible()`:
    /// `selectedResultIDs' = ids(R)`.
    /// Post after `clearSelection()`:
    /// `selectedResultIDs'' = ∅`.
    @Test
    func bulkSelectionTracksExactlyTheVisibleResults() {
        let state = UsageState(results: sampleResults)

        state.selectAllVisible()
        #expect(state.selectedResultIDs == Set(sampleResults.map(\.id)))

        state.clearSelection()
        #expect(state.selectedResultIDs.isEmpty)
    }

    private var sampleResults: [SearchAsset] {
        [
            SearchAsset(assetID: 1, absolutePath: "/tmp/library/one.jpg", thumbnailPath: nil),
            SearchAsset(assetID: 2, absolutePath: "/tmp/library/two.jpg", thumbnailPath: nil),
        ]
    }
}
