import Foundation

public extension InstallStep {
    var accessibilityKey: String {
        switch self {
        case .prepareDirectories:
            return "prepare-directories"
        case .prepareDatabase:
            return "prepare-database"
        case .downloadBaseModel:
            return "download-base-model"
        case .downloadStage1Checkpoint:
            return "download-stage1-checkpoint"
        case .downloadPublicAnchor:
            return "download-public-anchor"
        case .verifyArtifacts:
            return "verify-artifacts"
        case .finalizeInstallation:
            return "finalize-installation"
        }
    }

    var title: String {
        switch self {
        case .prepareDirectories:
            return "Prepare Directories"
        case .prepareDatabase:
            return "Prepare Database"
        case .downloadBaseModel:
            return "Download Shared Config"
        case .downloadStage1Checkpoint:
            return "Download Published Encoder"
        case .downloadPublicAnchor:
            return "Download Public Anchor"
        case .verifyArtifacts:
            return "Verify Artifacts"
        case .finalizeInstallation:
            return "Finalize Installation"
        }
    }

    var pendingMessage: String {
        switch self {
        case .prepareDirectories:
            return "Waiting to prepare local app directories"
        case .prepareDatabase:
            return "Waiting to prepare the local search database"
        case .downloadBaseModel:
            return "Waiting to fetch the shared tokenizer and config"
        case .downloadStage1Checkpoint:
            return "Waiting to fetch the published Lucas encoder"
        case .downloadPublicAnchor:
            return "Waiting to fetch the public adaptation anchor"
        case .verifyArtifacts:
            return "Waiting to verify the installed artifacts"
        case .finalizeInstallation:
            return "Waiting to finish installation"
        }
    }
}

extension InstallCoordinator {
    func resetIncompleteInstallationIfNeeded() throws {
        let fileManager = FileManager.default
        let installationVerifier = InstallationVerifier(paths: paths, catalog: catalog)
        guard try installationVerifier.isInstallationComplete() == false else {
            return
        }

        if fileManager.fileExists(atPath: paths.supportRoot.path) {
            try fileManager.removeItem(at: paths.supportRoot)
        }
    }

    var totalInstallUnits: Int {
        4
            + catalog.baseModel.requiredFiles.count
            + catalog.stage1Checkpoint.requiredFiles.count
            + catalog.publicAnchor.requiredFiles.count
    }

    func downloadProgress(
        step: InstallStep,
        artifactName: String,
        artifactProgress: ArtifactTransferProgress,
        completedUnitsBeforeStep: Int,
        totalUnits: Int,
        startedAt: Date
    ) -> InstallProgress {
        let completedUnits = Double(completedUnitsBeforeStep) + artifactProgress.fractionalCompletedFileCount
        let detail: String
        if artifactProgress.reusedExistingFile {
            detail = "Using \(artifactProgress.completedFileCount) of \(artifactProgress.totalFileCount) files"
        } else if artifactProgress.completedFileCount < artifactProgress.totalFileCount {
            let fileIndex = artifactProgress.completedFileCount + 1
            if let receivedBytes = artifactProgress.receivedBytes,
               let expectedBytes = artifactProgress.expectedBytes,
               expectedBytes > 0 {
                let percentage = Int((artifactProgress.currentFileFraction * 100).rounded())
                detail = "Downloading \(fileIndex) of \(artifactProgress.totalFileCount) files · \(percentage)% · \(format(bytes: receivedBytes)) of \(format(bytes: expectedBytes))"
            } else if let receivedBytes = artifactProgress.receivedBytes, receivedBytes > 0 {
                detail = "Downloading \(fileIndex) of \(artifactProgress.totalFileCount) files · \(format(bytes: receivedBytes)) received"
            } else {
                detail = "Preparing \(fileIndex) of \(artifactProgress.totalFileCount) files"
            }
        } else {
            detail = "Downloaded \(artifactProgress.completedFileCount) of \(artifactProgress.totalFileCount) files"
        }
        return timedProgress(
            step: step,
            message: "\(artifactName) · \(detail) · \(artifactProgress.filename)",
            stepProgress: artifactProgress.fractionalCompletedFileCount / Double(artifactProgress.totalFileCount),
            completedUnits: completedUnits,
            totalUnits: totalUnits,
            startedAt: startedAt,
            totalStepUnits: artifactProgress.totalFileCount,
            completedStepUnits: artifactProgress.fractionalCompletedFileCount
        )
    }

    func timedProgress(
        step: InstallStep,
        message: String,
        stepProgress: Double,
        completedUnits: Double,
        totalUnits: Int,
        startedAt: Date,
        totalStepUnits: Int? = nil,
        completedStepUnits: Double? = nil,
        remainingSeconds: Int? = nil
    ) -> InstallProgress {
        let elapsed = Int(Date().timeIntervalSince(startedAt))
        let resolvedRemaining: Int?
        if let remainingSeconds {
            resolvedRemaining = remainingSeconds
        } else if let totalStepUnits, let completedStepUnits, completedStepUnits > 0 {
            let remainingUnits = max(0, Double(totalStepUnits) - completedStepUnits)
            resolvedRemaining = Int((Double(elapsed) / completedStepUnits) * remainingUnits)
        } else {
            resolvedRemaining = nil
        }

        return InstallProgress(
            step: step,
            message: message,
            stepProgress: stepProgress,
            overallProgress: completedUnits / Double(totalUnits),
            elapsedSeconds: elapsed,
            remainingSeconds: resolvedRemaining,
            recordedAt: Date()
        )
    }

    func format(bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

actor InstallProgressBuffer {
    private var items: [InstallProgress]
    private let onProgress: (@Sendable (InstallProgress) async -> Void)?

    init(
        items: [InstallProgress],
        onProgress: (@Sendable (InstallProgress) async -> Void)?
    ) {
        self.items = items
        self.onProgress = onProgress
    }

    func apply(_ item: InstallProgress) async {
        if let index = items.lastIndex(where: { $0.step == item.step }) {
            items[index] = item
        } else {
            items.append(item)
        }
        if let onProgress {
            await onProgress(item)
        }
    }

    func snapshot() -> [InstallProgress] {
        items
    }
}
