import Foundation

@MainActor
public final class SelectionState {
    public private(set) var selectedAssetIDs: Set<Int64> = []

    public init() {}

    public func toggle(assetID: Int64) {
        if selectedAssetIDs.contains(assetID) {
            selectedAssetIDs.remove(assetID)
        } else {
            selectedAssetIDs.insert(assetID)
        }
    }

    public func selectAll(items: [SearchAssetRecord]) {
        selectedAssetIDs = Set(items.map(\.id))
    }

    public func clear() {
        selectedAssetIDs.removeAll()
    }
}
