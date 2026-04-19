import Foundation
import SemanticGalleryCore

struct InstallProgressPresentation {
    let progress: [InstallProgress]
    let now: Date

    var overallProgress: Double {
        InstallStep.allCases
            .map { liveDisplay(for: progress(for: $0)).progress * stepWeight(for: $0) }
            .reduce(0, +)
    }

    var currentProgressItem: InstallProgress {
        if let active = InstallStep.allCases
            .compactMap(existingProgress(for:))
            .filter({ liveDisplay(for: $0).progress < 1 })
            .max(by: { ($0.recordedAt ?? .distantPast) < ($1.recordedAt ?? .distantPast) }) {
            return active
        }

        if let completed = InstallStep.allCases
            .reversed()
            .compactMap(existingProgress(for:))
            .first(where: { liveDisplay(for: $0).progress >= 1 }) {
            return completed
        }

        return progress(for: .prepareDirectories)
    }

    func currentStepIndex() -> Int {
        (InstallStep.allCases.firstIndex(of: currentProgressItem.step) ?? 0) + 1
    }

    func progress(for step: InstallStep) -> InstallProgress {
        existingProgress(for: step) ?? placeholder(for: step)
    }

    private func existingProgress(for step: InstallStep) -> InstallProgress? {
        progress.last(where: { $0.step == step })
    }

    private func placeholder(for step: InstallStep) -> InstallProgress {
        InstallProgress(
            step: step,
            message: step.pendingMessage,
            stepProgress: 0,
            overallProgress: fallbackOverallProgress(before: step)
        )
    }

    private func fallbackOverallProgress(before step: InstallStep) -> Double {
        InstallStep.allCases
            .prefix { $0 != step }
            .compactMap(existingProgress(for:))
            .map { liveDisplay(for: $0).progress * stepWeight(for: $0.step) }
            .reduce(0, +)
    }

    private func liveDisplay(for item: InstallProgress) -> LiveProgressDisplay {
        LiveProgressPresentation(
            progress: item.stepProgress,
            elapsedSeconds: item.elapsedSeconds,
            remainingSeconds: item.remainingSeconds,
            recordedAt: item.recordedAt
        )
        .displayed(at: now)
    }

    private func stepWeight(for step: InstallStep) -> Double {
        let totalUnits = Double(
            4
                + ArtifactCatalog.legacyCompatible.baseModel.requiredFiles.count
                + ArtifactCatalog.legacyCompatible.stage1Checkpoint.requiredFiles.count
                + ArtifactCatalog.legacyCompatible.publicAnchor.requiredFiles.count
        )
        let units: Double
        switch step {
        case .prepareDirectories, .prepareDatabase, .verifyArtifacts, .finalizeInstallation:
            units = 1
        case .downloadBaseModel:
            units = Double(ArtifactCatalog.legacyCompatible.baseModel.requiredFiles.count)
        case .downloadStage1Checkpoint:
            units = Double(ArtifactCatalog.legacyCompatible.stage1Checkpoint.requiredFiles.count)
        case .downloadPublicAnchor:
            units = Double(ArtifactCatalog.legacyCompatible.publicAnchor.requiredFiles.count)
        }
        return units / totalUnits
    }
}
