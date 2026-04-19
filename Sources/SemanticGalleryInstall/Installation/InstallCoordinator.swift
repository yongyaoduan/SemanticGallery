import Foundation
import SemanticGalleryPersistence

public actor InstallCoordinator {
    let paths: AppPaths
    let downloader: ArtifactDownloading
    let catalog: ArtifactCatalog

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

        let installationVerifier = InstallationVerifier(paths: paths, catalog: catalog)
        guard try installationVerifier.isInstallationComplete() else {
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
}
