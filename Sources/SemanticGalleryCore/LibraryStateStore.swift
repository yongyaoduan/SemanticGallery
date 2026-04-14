import Foundation
import Observation

public enum FolderPreparationStep: Equatable, Sendable {
    case requestFolderAccess
    case scanSupportedImages
    case persistBookmark
    case finalizeFolderSelection
}

public struct FolderPreparationProgress: Equatable, Sendable {
    public let step: FolderPreparationStep
    public let message: String
    public let progress: Double

    public init(step: FolderPreparationStep, message: String, progress: Double) {
        self.step = step
        self.message = message
        self.progress = progress
    }
}

@MainActor
@Observable
public final class LibraryStateStore {
    public var selectedFolder: URL?
    public var folderPreparationProgress: [FolderPreparationProgress]

    public init(
        selectedFolder: URL? = nil,
        folderPreparationProgress: [FolderPreparationProgress] = []
    ) {
        self.selectedFolder = selectedFolder
        self.folderPreparationProgress = folderPreparationProgress
    }
}
