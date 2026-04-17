import Foundation
import Testing
@testable import SemanticGalleryCore

@Test
func liveProgressPresentationAdvancesElapsedAndRemainingBetweenSnapshots() {
    let recordedAt = Date(timeIntervalSince1970: 1_000)
    let presentation = LiveProgressPresentation(
        progress: 0.4,
        elapsedSeconds: 8,
        remainingSeconds: 12,
        recordedAt: recordedAt
    )

    let displayed = presentation.displayed(at: recordedAt.addingTimeInterval(3.2))

    #expect(displayed.elapsedSeconds == 11)
    #expect(displayed.remainingSeconds == 9)
    #expect(displayed.progress > 0.4)
    #expect(displayed.progress < 1.0)
}

@Test
func liveProgressPresentationKeepsCompletedStepsStable() {
    let recordedAt = Date(timeIntervalSince1970: 1_000)
    let presentation = LiveProgressPresentation(
        progress: 1.0,
        elapsedSeconds: 28,
        remainingSeconds: 0,
        recordedAt: recordedAt
    )

    let displayed = presentation.displayed(at: recordedAt.addingTimeInterval(5))

    #expect(displayed.elapsedSeconds == 28)
    #expect(displayed.remainingSeconds == 0)
    #expect(displayed.progress == 1.0)
}

@Test
func liveProgressPresentationAdvancesElapsedEvenWithoutRemainingEstimate() {
    let recordedAt = Date(timeIntervalSince1970: 1_000)
    let presentation = LiveProgressPresentation(
        progress: 0.0,
        elapsedSeconds: 0,
        remainingSeconds: nil,
        recordedAt: recordedAt
    )

    let displayed = presentation.displayed(at: recordedAt.addingTimeInterval(4.6))

    #expect(displayed.elapsedSeconds == 4)
    #expect(displayed.remainingSeconds == nil)
    #expect(displayed.progress == 0.0)
}
