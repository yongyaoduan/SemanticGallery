import Foundation
import Testing
@testable import SemanticGalleryCore

@Suite("Status Activity Contracts")
struct StatusActivityContractTests {
    /// Formal specification for callers:
    /// Pre: `progress ∈ ℝ`, `elapsedSeconds ∈ ℤ`, and `remainingSeconds ∈ ℤ ∪ {nil}`.
    /// Post after `StatusActivitySnapshot(...)`:
    /// `progress' ∈ [0, 1]`, `elapsedSeconds' ≥ 0`, and
    /// `remainingSeconds' = nil ∨ remainingSeconds' ≥ 0`.
    @Test
    func activitySnapshotClampsEveryCallerSuppliedScalarToTheVisibleRange() {
        let snapshot = StatusActivitySnapshot(
            progress: 4.0,
            elapsedSeconds: -12,
            remainingSeconds: -8,
            estimatedCompletionDate: nil
        )

        #expect(snapshot.progress == 1.0)
        #expect(snapshot.elapsedSeconds == 0)
        #expect(snapshot.remainingSeconds == 0)
    }

    /// Formal specification for callers:
    /// Pre: a folder-preparation progress history contains multiple entries for the same step.
    /// Post after `folderPreparation(history, now)`:
    /// the activity summary is derived from the latest visible snapshot for each step, not from older superseded entries.
    @Test
    func folderPreparationActivityUsesTheLatestCallerVisibleStepSnapshot() {
        let now = Date(timeIntervalSince1970: 10_000)
        let activity = StatusActivityMeter.folderPreparation([
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Earlier",
                stepProgress: 0.1,
                overallProgress: 0.1,
                elapsedSeconds: 3,
                remainingSeconds: 90,
                recordedAt: now.addingTimeInterval(-20)
            ),
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Latest",
                stepProgress: 0.75,
                overallProgress: 0.75,
                elapsedSeconds: 30,
                remainingSeconds: 10,
                recordedAt: now.addingTimeInterval(-5)
            ),
        ], now: now)

        #expect(activity?.progress == StatusProgressMeter.folderPreparation([
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Earlier",
                stepProgress: 0.1,
                overallProgress: 0.1,
                elapsedSeconds: 3,
                remainingSeconds: 90,
                recordedAt: now.addingTimeInterval(-20)
            ),
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Latest",
                stepProgress: 0.75,
                overallProgress: 0.75,
                elapsedSeconds: 30,
                remainingSeconds: 10,
                recordedAt: now.addingTimeInterval(-5)
            ),
        ]))
        #expect(activity?.elapsedSeconds == 35)
        #expect(activity?.remainingSeconds == 5)
    }
}
