import Foundation
import Observation

@MainActor
@Observable
public final class UsageState {
    public var queryText: String
    public var resultLimit: Int
    public var results: [SearchAsset]
    public var pastedImageData: Data?
    public var selectedResultIDs: Set<Int64>
    public var isSearching: Bool
    public var isSelectionModeEnabled: Bool
    public var previewResultID: Int64?
    public var isPreviewMetadataVisible: Bool

    public init(
        queryText: String = "",
        resultLimit: Int = 50,
        results: [SearchAsset] = [],
        pastedImageData: Data? = nil,
        selectedResultIDs: Set<Int64> = [],
        isSearching: Bool = false,
        isSelectionModeEnabled: Bool = false,
        previewResultID: Int64? = nil,
        isPreviewMetadataVisible: Bool = false
    ) {
        self.queryText = queryText
        self.resultLimit = resultLimit
        self.results = results
        self.pastedImageData = pastedImageData
        self.selectedResultIDs = selectedResultIDs
        self.isSearching = isSearching
        self.isSelectionModeEnabled = isSelectionModeEnabled
        self.previewResultID = previewResultID
        self.isPreviewMetadataVisible = isPreviewMetadataVisible
    }

    public var selectionCount: Int {
        selectedResultIDs.count
    }

    public func isSelected(_ resultID: Int64) -> Bool {
        selectedResultIDs.contains(resultID)
    }

    public func toggleSelection(resultID: Int64) {
        if selectedResultIDs.contains(resultID) {
            selectedResultIDs.remove(resultID)
        } else {
            selectedResultIDs.insert(resultID)
        }
    }

    public func selectAllVisible() {
        selectedResultIDs = Set(results.map(\.id))
    }

    public func clearSelection() {
        selectedResultIDs.removeAll()
    }

    public func enterSelectionMode() {
        isSelectionModeEnabled = true
        clearSelection()
        closePreview()
    }

    public func leaveSelectionMode() {
        isSelectionModeEnabled = false
        clearSelection()
    }

    public func openPreview(resultID: Int64) {
        isSelectionModeEnabled = false
        clearSelection()
        previewResultID = resultID
        isPreviewMetadataVisible = false
    }

    public func closePreview() {
        previewResultID = nil
        isPreviewMetadataVisible = false
    }

    public func togglePreviewMetadata() {
        guard previewResultID != nil else {
            isPreviewMetadataVisible = false
            return
        }
        isPreviewMetadataVisible.toggle()
    }

    public func showNextPreviewItem() {
        movePreview(by: 1)
    }

    public func showPreviousPreviewItem() {
        movePreview(by: -1)
    }

    public var previewItem: SearchAsset? {
        guard let previewResultID else {
            return nil
        }
        return results.first(where: { $0.id == previewResultID })
    }

    private func movePreview(by offset: Int) {
        guard
            let previewResultID,
            let currentIndex = results.firstIndex(where: { $0.id == previewResultID })
        else {
            return
        }

        let nextIndex = currentIndex + offset
        guard results.indices.contains(nextIndex) else {
            return
        }

        self.previewResultID = results[nextIndex].id
        isPreviewMetadataVisible = false
    }
}
