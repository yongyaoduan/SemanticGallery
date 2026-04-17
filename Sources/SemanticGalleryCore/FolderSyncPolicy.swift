public enum FolderSyncPolicy {
    public static func matrixRefreshThreshold(forVisibleImageCount count: Int) -> Int {
        Int((Double(max(count, 0)) * 0.1).rounded(.down))
    }

    public static func shouldRebuildActiveView(
        currentViewExists: Bool,
        pendingChangeCount: Int,
        visibleImageCount: Int
    ) -> Bool {
        if currentViewExists == false {
            return true
        }
        return pendingChangeCount >= matrixRefreshThreshold(forVisibleImageCount: visibleImageCount)
    }
}
