public enum StatusTitleFormatter {
    public static func title(
        for status: AppStatus,
        selectedFolderExists: Bool,
        supportedImageCount: Int?,
        searchableImageCount: Int?
    ) -> String {
        switch status {
        case .indexing:
            return "Indexing"
        case .training:
            return "Adapting"
        case .readyWithoutFolder, .installRequired:
            return "No folder selected"
        case .searching, .ready, .installing, .installFailed, .installComplete, .uninstalling:
            guard selectedFolderExists else {
                return "No folder selected"
            }

            let supportedCount = max(0, supportedImageCount ?? 0)
            let searchableCount = max(0, searchableImageCount ?? supportedCount)

            if searchableCount < supportedCount {
                let noun = supportedCount == 1 ? "image" : "images"
                return "\(searchableCount) of \(supportedCount) \(noun) ready"
            }

            let noun = searchableCount == 1 ? "image" : "images"
            return "\(searchableCount) \(noun) ready"
        }
    }
}
