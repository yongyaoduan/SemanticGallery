import Foundation

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
