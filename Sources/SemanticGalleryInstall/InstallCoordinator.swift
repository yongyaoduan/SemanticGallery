import Foundation
import Observation
import SemanticGalleryPersistence

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
public final class InstallationStateStore {
    public var progress: [InstallProgress]

    public init(progress: [InstallProgress] = []) {
        self.progress = progress
    }
}

public actor InstallCoordinator {
    private let paths: AppPaths
    private let downloader: ArtifactDownloading
    private let catalog: ArtifactCatalog

    public init(
        paths: AppPaths,
        downloader: ArtifactDownloading,
        catalog: ArtifactCatalog = .legacyCompatible
    ) {
        self.paths = paths
        self.downloader = downloader
        self.catalog = catalog
    }

    public func prepare(
        onProgress: (@Sendable (InstallProgress) async -> Void)? = nil
    ) async throws -> [InstallProgress] {
        try resetIncompleteInstallationIfNeeded()

        let progressBuffer = InstallProgressBuffer(
            items: InstallProgress.placeholders(),
            onProgress: onProgress
        )
        let totalUnits = totalInstallUnits
        var completedUnits = 0

        let directoriesStart = Date()
        try FileManager.default.createDirectory(at: paths.supportRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: paths.artifactsRoot, withIntermediateDirectories: true)
        completedUnits += 1
        await progressBuffer.apply(
            timedProgress(
                step: .prepareDirectories,
                message: "Prepared app support, cache, and artifact directories",
                stepProgress: 1.0,
                completedUnits: Double(completedUnits),
                totalUnits: totalUnits,
                startedAt: directoriesStart,
                remainingSeconds: 0
            )
        )

        let databaseStart = Date()
        if FileManager.default.fileExists(atPath: paths.databaseURL.path) == false {
            FileManager.default.createFile(atPath: paths.databaseURL.path, contents: Data())
        }
        completedUnits += 1
        await progressBuffer.apply(
            timedProgress(
                step: .prepareDatabase,
                message: "Prepared the local search database",
                stepProgress: 1.0,
                completedUnits: Double(completedUnits),
                totalUnits: totalUnits,
                startedAt: databaseStart,
                remainingSeconds: 0
            )
        )

        let baseModelStart = Date()
        let baseModelCompletedUnits = completedUnits
        await progressBuffer.apply(
            timedProgress(
                step: .downloadBaseModel,
                message: "Preparing the shared config, preprocessor, and tokenizer files",
                stepProgress: 0,
                completedUnits: Double(baseModelCompletedUnits),
                totalUnits: totalUnits,
                startedAt: baseModelStart,
                totalStepUnits: catalog.baseModel.requiredFiles.count,
                completedStepUnits: 0
            )
        )
        try await downloader.download(artifact: catalog.baseModel, into: paths.artifactsRoot) { artifactProgress in
            await progressBuffer.apply(
                self.downloadProgress(
                    step: .downloadBaseModel,
                    artifactName: "Shared SigLIP2 assets",
                    artifactProgress: artifactProgress,
                    completedUnitsBeforeStep: baseModelCompletedUnits,
                    totalUnits: totalUnits,
                    startedAt: baseModelStart
                )
            )
        }
        completedUnits += catalog.baseModel.requiredFiles.count

        let stage1Start = Date()
        let stage1CompletedUnits = completedUnits
        await progressBuffer.apply(
            timedProgress(
                step: .downloadStage1Checkpoint,
                message: "Preparing the published Lucas encoder weights",
                stepProgress: 0,
                completedUnits: Double(stage1CompletedUnits),
                totalUnits: totalUnits,
                startedAt: stage1Start,
                totalStepUnits: catalog.stage1Checkpoint.requiredFiles.count,
                completedStepUnits: 0
            )
        )
        try await downloader.download(artifact: catalog.stage1Checkpoint, into: paths.artifactsRoot) { artifactProgress in
            await progressBuffer.apply(
                self.downloadProgress(
                    step: .downloadStage1Checkpoint,
                    artifactName: "Published Lucas encoder",
                    artifactProgress: artifactProgress,
                    completedUnitsBeforeStep: stage1CompletedUnits,
                    totalUnits: totalUnits,
                    startedAt: stage1Start
                )
            )
        }
        completedUnits += catalog.stage1Checkpoint.requiredFiles.count

        let publicAnchorStart = Date()
        let publicAnchorCompletedUnits = completedUnits
        await progressBuffer.apply(
            timedProgress(
                step: .downloadPublicAnchor,
                message: "Preparing the shared anchor set",
                stepProgress: 0,
                completedUnits: Double(publicAnchorCompletedUnits),
                totalUnits: totalUnits,
                startedAt: publicAnchorStart,
                totalStepUnits: catalog.publicAnchor.requiredFiles.count,
                completedStepUnits: 0
            )
        )
        try await downloader.download(artifact: catalog.publicAnchor, into: paths.artifactsRoot) { artifactProgress in
            await progressBuffer.apply(
                self.downloadProgress(
                    step: .downloadPublicAnchor,
                    artifactName: "Public adaptation anchor",
                    artifactProgress: artifactProgress,
                    completedUnitsBeforeStep: publicAnchorCompletedUnits,
                    totalUnits: totalUnits,
                    startedAt: publicAnchorStart
                )
            )
        }
        completedUnits += catalog.publicAnchor.requiredFiles.count

        let verificationStart = Date()
        try Data(#"{"status":"complete"}"#.utf8).write(to: paths.installStateURL)
        completedUnits += 1
        await progressBuffer.apply(
            timedProgress(
                step: .verifyArtifacts,
                message: "Verified model files, checkpoint weights, and anchor data",
                stepProgress: 1.0,
                completedUnits: Double(completedUnits),
                totalUnits: totalUnits,
                startedAt: verificationStart,
                remainingSeconds: 0
            )
        )

        let installStateStore = InstallStateStore(paths: paths, catalog: catalog)
        guard try installStateStore.isInstallationComplete() else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let finalizeStart = Date()
        completedUnits += 1
        await progressBuffer.apply(
            timedProgress(
                step: .finalizeInstallation,
                message: "SemanticGallery is ready to enter",
                stepProgress: 1.0,
                completedUnits: Double(completedUnits),
                totalUnits: totalUnits,
                startedAt: finalizeStart,
                remainingSeconds: 0
            )
        )
        return await progressBuffer.snapshot()
    }

    private func resetIncompleteInstallationIfNeeded() throws {
        let fileManager = FileManager.default
        let installStateStore = InstallStateStore(paths: paths, catalog: catalog)
        guard try installStateStore.isInstallationComplete() == false else {
            return
        }

        if fileManager.fileExists(atPath: paths.supportRoot.path) {
            try fileManager.removeItem(at: paths.supportRoot)
        }
    }

    private var totalInstallUnits: Int {
        4
            + catalog.baseModel.requiredFiles.count
            + catalog.stage1Checkpoint.requiredFiles.count
            + catalog.publicAnchor.requiredFiles.count
    }

    private func downloadProgress(
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

    private func timedProgress(
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

    private func format(bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

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

private actor InstallProgressBuffer {
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
