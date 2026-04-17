import Foundation

public struct LiveProgressDisplay: Equatable, Sendable {
    public let progress: Double
    public let elapsedSeconds: Int?
    public let remainingSeconds: Int?

    public init(progress: Double, elapsedSeconds: Int?, remainingSeconds: Int?) {
        self.progress = min(max(progress, 0), 1)
        self.elapsedSeconds = elapsedSeconds
        self.remainingSeconds = remainingSeconds
    }
}

public struct LiveProgressPresentation: Equatable, Sendable {
    public let progress: Double
    public let elapsedSeconds: Int?
    public let remainingSeconds: Int?
    public let recordedAt: Date?

    public init(
        progress: Double,
        elapsedSeconds: Int?,
        remainingSeconds: Int?,
        recordedAt: Date?
    ) {
        self.progress = min(max(progress, 0), 1)
        self.elapsedSeconds = elapsedSeconds
        self.remainingSeconds = remainingSeconds
        self.recordedAt = recordedAt
    }

    public func displayed(at now: Date = .now) -> LiveProgressDisplay {
        guard
            progress < 1,
            let recordedAt,
            now >= recordedAt
        else {
            return LiveProgressDisplay(
                progress: progress,
                elapsedSeconds: elapsedSeconds,
                remainingSeconds: remainingSeconds
            )
        }

        let delta = Int(now.timeIntervalSince(recordedAt).rounded(.down))
        let displayedElapsed = elapsedSeconds.map { max($0, $0 + delta) }
        let displayedRemaining = remainingSeconds.map { max(0, $0 - delta) }
        let displayedProgress: Double

        if let displayedElapsed, let displayedRemaining {
            let total = max(1, displayedElapsed + displayedRemaining)
            displayedProgress = min(0.999, max(progress, Double(displayedElapsed) / Double(total)))
        } else {
            displayedProgress = progress
        }

        return LiveProgressDisplay(
            progress: displayedProgress,
            elapsedSeconds: displayedElapsed,
            remainingSeconds: displayedRemaining
        )
    }
}
