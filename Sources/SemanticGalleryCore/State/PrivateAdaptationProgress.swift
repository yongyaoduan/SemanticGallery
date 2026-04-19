import Foundation

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
