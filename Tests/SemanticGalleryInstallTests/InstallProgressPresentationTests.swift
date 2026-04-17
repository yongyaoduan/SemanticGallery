import Foundation
import Testing
@testable import SemanticGalleryInstall

@Test
func installProgressPresentationHandlesMissingProgressWithoutRecursion() {
    let presentation = InstallProgressPresentation(progress: [], now: Date(timeIntervalSince1970: 1_000))

    #expect(presentation.overallProgress == 0)
    #expect(presentation.currentProgressItem.step == .prepareDirectories)
    #expect(presentation.progress(for: .prepareDirectories).message == InstallStep.prepareDirectories.pendingMessage)
}

@Test
func installProgressPresentationCombinesSparseProgressUsingStepWeights() {
    let now = Date(timeIntervalSince1970: 1_000)
    let progress = [
        InstallProgress(
            step: .prepareDirectories,
            message: "Prepared",
            stepProgress: 1.0,
            overallProgress: 1.0 / 13.0,
            elapsedSeconds: 1,
            remainingSeconds: 0,
            recordedAt: now
        ),
        InstallProgress(
            step: .downloadBaseModel,
            message: "Downloading",
            stepProgress: 0.4,
            overallProgress: 3.0 / 13.0,
            elapsedSeconds: 4,
            remainingSeconds: 6,
            recordedAt: now
        ),
    ]
    let presentation = InstallProgressPresentation(progress: progress, now: now)

    #expect(abs(presentation.overallProgress - (3.0 / 13.0)) < 0.0001)
    #expect(presentation.currentProgressItem.step == .downloadBaseModel)
}
