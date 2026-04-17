import Foundation
import Observation

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

public enum PrivateAdaptationStep: Equatable, Sendable, CaseIterable {
    case prepareData
    case trainModel
    case reindexLibrary
}

public struct PrivateAdaptationProgress: Equatable, Sendable {
    public let step: PrivateAdaptationStep
    public let message: String
    public let stepProgress: Double
    public let overallProgress: Double
    public let elapsedSeconds: Int?
    public let remainingSeconds: Int?
    public let recordedAt: Date?

    public init(
        step: PrivateAdaptationStep,
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

    public static func placeholders() -> [PrivateAdaptationProgress] {
        PrivateAdaptationStep.allCases.map { step in
            PrivateAdaptationProgress(
                step: step,
                message: step.pendingMessage,
                stepProgress: 0,
                overallProgress: 0
            )
        }
    }
}

@MainActor
@Observable
public final class LibraryStateStore {
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

public extension PrivateAdaptationStep {
    var statusWeight: Double {
        switch self {
        case .prepareData, .trainModel, .reindexLibrary:
            return 1.0 / 3.0
        }
    }

    var accessibilityKey: String {
        switch self {
        case .prepareData:
            return "prepare-data"
        case .trainModel:
            return "train-model"
        case .reindexLibrary:
            return "reindex-library"
        }
    }

    var title: String {
        switch self {
        case .prepareData:
            return "Data Preparation"
        case .trainModel:
            return "Model Training"
        case .reindexLibrary:
            return "Reindexing"
        }
    }

    var pendingMessage: String {
        switch self {
        case .prepareData:
            return "Waiting to prepare adaptation data"
        case .trainModel:
            return "Waiting to begin local training"
        case .reindexLibrary:
            return "Waiting to rebuild the search index"
        }
    }
}

public enum StatusProgressMeter {
    public static func folderPreparation(_ progressItems: [FolderPreparationProgress]) -> Double {
        weightedProgress(
            latestProgressByStep: latestFolderPreparationProgressByStep(from: progressItems),
            steps: FolderPreparationStep.allCases,
            weight: \.statusWeight
        )
    }

    public static func privateAdaptation(_ progressItems: [PrivateAdaptationProgress]) -> Double {
        weightedProgress(
            latestProgressByStep: latestPrivateAdaptationProgressByStep(from: progressItems),
            steps: PrivateAdaptationStep.allCases,
            weight: \.statusWeight
        )
    }

    private static func latestFolderPreparationProgressByStep(
        from items: [FolderPreparationProgress]
    ) -> [FolderPreparationStep: Double] {
        Dictionary(
            uniqueKeysWithValues: FolderPreparationStep.allCases.map { step in
                let latest = items.last(where: { $0.step == step })?.stepProgress ?? 0
                return (step, latest)
            }
        )
    }

    private static func latestPrivateAdaptationProgressByStep(
        from items: [PrivateAdaptationProgress]
    ) -> [PrivateAdaptationStep: Double] {
        Dictionary(
            uniqueKeysWithValues: PrivateAdaptationStep.allCases.map { step in
                let latest = items.last(where: { $0.step == step })?.stepProgress ?? 0
                return (step, latest)
            }
        )
    }

    private static func weightedProgress<Step: Hashable>(
        latestProgressByStep: [Step: Double],
        steps: [Step],
        weight: KeyPath<Step, Double>
    ) -> Double {
        let total = steps.reduce(into: 0.0) { value, step in
            let boundedStepProgress = min(max(latestProgressByStep[step] ?? 0, 0), 1)
            value += boundedStepProgress * step[keyPath: weight]
        }
        return min(max(total, 0), 1)
    }
}

public struct StatusActivitySnapshot: Equatable, Sendable {
    public let progress: Double
    public let elapsedSeconds: Int
    public let remainingSeconds: Int?
    public let estimatedCompletionDate: Date?

    public init(
        progress: Double,
        elapsedSeconds: Int,
        remainingSeconds: Int?,
        estimatedCompletionDate: Date?
    ) {
        self.progress = min(max(progress, 0), 1)
        self.elapsedSeconds = max(0, elapsedSeconds)
        self.remainingSeconds = remainingSeconds.map { max(0, $0) }
        self.estimatedCompletionDate = estimatedCompletionDate
    }
}

public enum StatusActivityMeter {
    public static func folderPreparation(
        _ progressItems: [FolderPreparationProgress],
        now: Date = .now
    ) -> StatusActivitySnapshot? {
        guard let latest = latestFolderPreparationProgressItem(from: progressItems) else {
            return nil
        }

        return snapshot(
            progress: StatusProgressMeter.folderPreparation(progressItems),
            elapsedSeconds: latest.elapsedSeconds,
            remainingSeconds: latest.remainingSeconds,
            recordedAt: latest.recordedAt,
            now: now
        )
    }

    public static func privateAdaptation(
        _ progressItems: [PrivateAdaptationProgress],
        now: Date = .now
    ) -> StatusActivitySnapshot? {
        guard let latest = latestPrivateAdaptationProgressItem(from: progressItems) else {
            return nil
        }

        return snapshot(
            progress: StatusProgressMeter.privateAdaptation(progressItems),
            elapsedSeconds: latest.elapsedSeconds,
            remainingSeconds: latest.remainingSeconds,
            recordedAt: latest.recordedAt,
            now: now
        )
    }

    private static func latestFolderPreparationProgressItem(
        from items: [FolderPreparationProgress]
    ) -> FolderPreparationProgress? {
        items.max(by: progressOrdering)
    }

    private static func latestPrivateAdaptationProgressItem(
        from items: [PrivateAdaptationProgress]
    ) -> PrivateAdaptationProgress? {
        items.max(by: progressOrdering)
    }

    private static func progressOrdering<ProgressItem: HasRecordedProgress>(
        _ left: ProgressItem,
        _ right: ProgressItem
    ) -> Bool {
        if let leftDate = left.recordedAt, let rightDate = right.recordedAt, leftDate != rightDate {
            return leftDate < rightDate
        }
        if left.recordedAt != nil || right.recordedAt != nil {
            return left.recordedAt == nil && right.recordedAt != nil
        }
        return left.progressValue < right.progressValue
    }

    private static func snapshot(
        progress: Double,
        elapsedSeconds: Int?,
        remainingSeconds: Int?,
        recordedAt: Date?,
        now: Date
    ) -> StatusActivitySnapshot {
        let liveProgress = LiveProgressPresentation(
            progress: progress,
            elapsedSeconds: elapsedSeconds,
            remainingSeconds: remainingSeconds,
            recordedAt: recordedAt
        ).displayed(at: now)
        let estimatedCompletionDate = liveProgress.remainingSeconds.map { seconds in
            now.addingTimeInterval(Double(seconds))
        }

        return StatusActivitySnapshot(
            progress: progress,
            elapsedSeconds: liveProgress.elapsedSeconds ?? 0,
            remainingSeconds: liveProgress.remainingSeconds,
            estimatedCompletionDate: estimatedCompletionDate
        )
    }
}

private protocol HasRecordedProgress {
    var recordedAt: Date? { get }
    var progressValue: Double { get }
}

extension FolderPreparationProgress: HasRecordedProgress {
    fileprivate var progressValue: Double {
        overallProgress
    }
}

extension PrivateAdaptationProgress: HasRecordedProgress {
    fileprivate var progressValue: Double {
        overallProgress
    }
}
