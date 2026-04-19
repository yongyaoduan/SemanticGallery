import Foundation
import Observation

@MainActor
@Observable
public final class LibraryState {
    public var selectedFolder: URL?
    public var folderPreparationProgress: [FolderPreparationProgress]
    public var privateAdaptationProgress: [PrivateAdaptationProgress]
    public var privateAdaptationNotice: String?
    public var supportedImageCount: Int?
    public var searchableImageCount: Int?
    public var activeEncoderVersion: String?
    public var searchIssue: String?

    public init(
        selectedFolder: URL? = nil,
        folderPreparationProgress: [FolderPreparationProgress] = [],
        privateAdaptationProgress: [PrivateAdaptationProgress] = [],
        privateAdaptationNotice: String? = nil,
        supportedImageCount: Int? = nil,
        searchableImageCount: Int? = nil,
        activeEncoderVersion: String? = nil,
        searchIssue: String? = nil
    ) {
        self.selectedFolder = selectedFolder
        self.folderPreparationProgress = folderPreparationProgress
        self.privateAdaptationProgress = privateAdaptationProgress
        self.privateAdaptationNotice = privateAdaptationNotice
        self.supportedImageCount = supportedImageCount
        self.searchableImageCount = searchableImageCount
        self.activeEncoderVersion = activeEncoderVersion
        self.searchIssue = searchIssue
    }
}
