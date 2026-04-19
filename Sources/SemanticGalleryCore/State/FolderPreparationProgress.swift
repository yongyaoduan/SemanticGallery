import Foundation

public enum FolderPreparationStep: Equatable, Sendable, CaseIterable {
    case requestFolderAccess
    case scanSupportedImages
    case reuseExistingEmbeddings
    case generateEmbeddings
    case buildSearchView
    case finalizeFolderSelection
}

public struct FolderPreparationProgress: Equatable, Sendable {
    public let step: FolderPreparationStep
    public let message: String
    public let stepProgress: Double
    public let overallProgress: Double
    public let elapsedSeconds: Int?
    public let remainingSeconds: Int?
    public let recordedAt: Date?

    public init(
        step: FolderPreparationStep,
        message: String,
        stepProgress: Double,
        overallProgress: Double,
        elapsedSeconds: Int? = nil,
        remainingSeconds: Int? = nil,
        recordedAt: Date? = nil
    ) {
        self.step = step
        self.message = message
        self.stepProgress = stepProgress
        self.overallProgress = overallProgress
        self.elapsedSeconds = elapsedSeconds
        self.remainingSeconds = remainingSeconds
        self.recordedAt = recordedAt
    }

    public var progress: Double {
        overallProgress
    }

    public static func placeholders() -> [FolderPreparationProgress] {
        FolderPreparationStep.allCases.map { step in
            FolderPreparationProgress(
                step: step,
                message: step.pendingMessage,
                stepProgress: 0,
                overallProgress: 0,
                elapsedSeconds: nil,
                remainingSeconds: nil
            )
        }
    }
}

public extension FolderPreparationStep {
    var statusWeight: Double {
        switch self {
        case .requestFolderAccess,
                .scanSupportedImages,
                .reuseExistingEmbeddings,
                .generateEmbeddings,
                .buildSearchView,
                .finalizeFolderSelection:
            return 1.0 / 6.0
        }
    }

    var accessibilityKey: String {
        switch self {
        case .requestFolderAccess:
            return "request-folder-access"
        case .scanSupportedImages:
            return "scan-supported-images"
        case .reuseExistingEmbeddings:
            return "reuse-existing-embeddings"
        case .generateEmbeddings:
            return "generate-embeddings"
        case .buildSearchView:
            return "build-search-view"
        case .finalizeFolderSelection:
            return "finalize-folder-selection"
        }
    }

    var title: String {
        switch self {
        case .requestFolderAccess:
            return "Authorize"
        case .scanSupportedImages:
            return "Scan Images"
        case .reuseExistingEmbeddings:
            return "Reuse Embeddings"
        case .generateEmbeddings:
            return "Generate Embeddings"
        case .buildSearchView:
            return "Build Search View"
        case .finalizeFolderSelection:
            return "Finalize"
        }
    }

    var pendingMessage: String {
        switch self {
        case .requestFolderAccess:
            return "Waiting for folder access"
        case .scanSupportedImages:
            return "Waiting to scan supported images"
        case .reuseExistingEmbeddings:
            return "Waiting to check reusable embeddings"
        case .generateEmbeddings:
            return "Waiting to generate missing embeddings"
        case .buildSearchView:
            return "Waiting to prepare the search view"
        case .finalizeFolderSelection:
            return "Waiting to finish setup"
        }
    }
}
