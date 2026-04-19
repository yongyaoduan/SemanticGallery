import Foundation
import Observation

public enum InstallStep: Equatable, Sendable, CaseIterable {
    case prepareDirectories
    case prepareDatabase
    case downloadBaseModel
    case downloadStage1Checkpoint
    case downloadPublicAnchor
    case verifyArtifacts
    case finalizeInstallation
}

public struct InstallProgress: Equatable, Sendable {
    public let step: InstallStep
    public let message: String
    public let stepProgress: Double
    public let overallProgress: Double
    public let elapsedSeconds: Int?
    public let remainingSeconds: Int?
    public let recordedAt: Date?

    public init(
        step: InstallStep,
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

    public static func placeholders() -> [InstallProgress] {
        InstallStep.allCases.map { step in
            InstallProgress(
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
public final class InstallationState {
    public var progress: [InstallProgress]

    public init(progress: [InstallProgress] = []) {
        self.progress = progress
    }
}
