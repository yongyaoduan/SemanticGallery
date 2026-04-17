import Foundation
import Observation

@MainActor
@Observable
public final class WorkspaceStateStore {
    public var queryText: String
    public var resultLimit: Int
    public var results: [SearchAssetRecord]
    public var pastedImageData: Data?
    public var selectedAssetIDs: Set<Int64>
    public var isSearching: Bool
    public var isSelectionModeEnabled: Bool
    public var previewAssetID: Int64?
    public var isPreviewMetadataVisible: Bool

    public init(
        queryText: String = "",
        resultLimit: Int = 50,
        results: [SearchAssetRecord] = [],
        pastedImageData: Data? = nil,
        selectedAssetIDs: Set<Int64> = [],
        isSearching: Bool = false,
        isSelectionModeEnabled: Bool = false,
        previewAssetID: Int64? = nil,
        isPreviewMetadataVisible: Bool = false
    ) {
        self.queryText = queryText
        self.resultLimit = resultLimit
        self.results = results
        self.pastedImageData = pastedImageData
        self.selectedAssetIDs = selectedAssetIDs
        self.isSearching = isSearching
        self.isSelectionModeEnabled = isSelectionModeEnabled
        self.previewAssetID = previewAssetID
        self.isPreviewMetadataVisible = isPreviewMetadataVisible
    }

    public var selectionCount: Int {
        selectedAssetIDs.count
    }

    public func isSelected(_ assetID: Int64) -> Bool {
        selectedAssetIDs.contains(assetID)
    }

    public func toggleSelection(assetID: Int64) {
        if selectedAssetIDs.contains(assetID) {
            selectedAssetIDs.remove(assetID)
        } else {
            selectedAssetIDs.insert(assetID)
        }
    }

    public func selectAllVisible() {
        selectedAssetIDs = Set(results.map(\.id))
    }

    public func clearSelection() {
        selectedAssetIDs.removeAll()
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

    public func openPreview(assetID: Int64) {
        isSelectionModeEnabled = false
        clearSelection()
        previewAssetID = assetID
        isPreviewMetadataVisible = false
    }

    public func closePreview() {
        previewAssetID = nil
        isPreviewMetadataVisible = false
    }

    public func togglePreviewMetadata() {
        guard previewAssetID != nil else {
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

    public var previewItem: SearchAssetRecord? {
        guard let previewAssetID else {
            return nil
        }
        return results.first(where: { $0.id == previewAssetID })
    }

    private func movePreview(by offset: Int) {
        guard
            let previewAssetID,
            let currentIndex = results.firstIndex(where: { $0.id == previewAssetID })
        else {
            return
        }

        let nextIndex = currentIndex + offset
        guard results.indices.contains(nextIndex) else {
            return
        }

        self.previewAssetID = results[nextIndex].id
        isPreviewMetadataVisible = false
    }
}
