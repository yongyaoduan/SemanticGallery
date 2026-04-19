import Foundation
import SemanticGalleryCore
import SemanticGalleryIndex
import SemanticGalleryML

extension GalleryController {
    public func startPrivateAdaptation() {
        guard let adaptationTrainer else {
            return
        }

        guard let selectedFolder = libraryState.selectedFolder else {
            libraryState.privateAdaptationNotice =
                "Choose a folder in Settings before starting private adaptation."
            libraryState.privateAdaptationProgress = []
            statusState.status = .readyWithoutFolder
            return
        }

        if libraryState.folderPreparationProgress.isEmpty == false {
            libraryState.privateAdaptationNotice =
                "Wait for indexing to finish before starting private adaptation."
            return
        }

        if statusState.status == .training {
            libraryState.privateAdaptationNotice =
                "Private adaptation is already running."
            return
        }

        let supportedImageCount = libraryState.supportedImageCount ?? 0
        guard supportedImageCount >= 100 else {
            libraryState.privateAdaptationNotice =
                "This folder currently has \(supportedImageCount) supported images. SemanticGallery needs at least 100 images before local adaptation can begin."
            libraryState.privateAdaptationProgress = []
            statusState.status = .ready
            return
        }

        libraryState.privateAdaptationNotice = nil
        libraryState.privateAdaptationProgress = PrivateAdaptationProgress.placeholders()
        statusState.status = .training

        let trainingRunID: Int64?
        do {
            let database = try ensureDatabase()
            let folderID = try database.upsertFolder(
                absolutePath: selectedFolder.path(percentEncoded: false),
                bookmarkData: nil,
                isActive: true
            )
            trainingRunID = try database.insertTrainingRun(
                folderID: folderID,
                encoderVersion: activeEncoderVersion,
                status: "running",
                summaryJSON: nil
            )
        } catch {
            trainingRunID = nil
        }

        Task {
            do {
                let adaptation = try await adaptationTrainer.runAdaptation(for: selectedFolder) { progress in
                    await MainActor.run {
                        self.replacePrivateAdaptationProgress(progress)
                    }
                }

                await MainActor.run {
                    self.activeEmbeddingService = adaptation.embeddingService
                    self.activeEncoderVersion = adaptation.artifact.encoderVersion
                    self.libraryState.activeEncoderVersion = adaptation.artifact.encoderVersion
                    self.cachedVectorQuery = nil
                }

                try await self.rebuildActiveEncoderIndex(for: selectedFolder)

                if let trainingRunID {
                    let database = try self.ensureDatabase()
                    let summaryJSON = try? String(contentsOf: adaptation.artifact.summaryURL, encoding: .utf8)
                    try database.finishTrainingRun(
                        id: trainingRunID,
                        status: "completed",
                        summaryJSON: summaryJSON
                    )
                }

                await MainActor.run {
                    self.statusState.status = .ready
                }
            } catch {
                if let trainingRunID, let database = try? self.ensureDatabase() {
                    try? database.finishTrainingRun(
                        id: trainingRunID,
                        status: "failed",
                        summaryJSON: nil
                    )
                }
                await MainActor.run {
                    self.libraryState.privateAdaptationProgress = []
                    self.libraryState.privateAdaptationNotice = "Private adaptation could not finish."
                    self.cachedVectorQuery = nil
                    self.statusState.status = .ready
                }
            }
        }
    }

    func replacePrivateAdaptationProgress(_ progress: PrivateAdaptationProgress) {
        if let index = libraryState.privateAdaptationProgress.lastIndex(where: { $0.step == progress.step }) {
            libraryState.privateAdaptationProgress[index] = progress
        } else {
            libraryState.privateAdaptationProgress.append(progress)
        }
    }

    func rebuildActiveEncoderIndex(for url: URL) async throws {
        let database = try ensureDatabase()
        let startDate = Date()

        await MainActor.run {
            self.replacePrivateAdaptationProgress(
                PrivateAdaptationProgress(
                    step: .reindexLibrary,
                    message: "Refreshing the folder index",
                    stepProgress: 0.0,
                    overallProgress: 2.0 / 3.0,
                    elapsedSeconds: 0,
                    remainingSeconds: nil
                )
            )
        }

        let indexSummary = try await FolderIndexer(database: database).rebuildIndex(for: url)
        let missingEmbeddings = try database.missingEmbeddingWork(
            encoderVersion: activeEncoderVersion
        )

        if missingEmbeddings.isEmpty {
            await MainActor.run {
                self.replacePrivateAdaptationProgress(
                    PrivateAdaptationProgress(
                        step: .reindexLibrary,
                        message: "Search index is already current",
                        stepProgress: 1.0,
                        overallProgress: 1.0,
                        elapsedSeconds: Int(Date().timeIntervalSince(startDate)),
                        remainingSeconds: 0
                    )
                )
            }
        } else {
            let skippedCount = try await storeEmbeddingsInBatches(
                missingEmbeddings,
                database: database,
                embeddingService: activeEmbeddingService,
                encoderVersion: activeEncoderVersion
            ) { completedCount, totalCount, skippedCount, elapsedSeconds, remainingSeconds in
                let completed = totalCount == 0 ? 1.0 : Double(completedCount) / Double(totalCount)
                let message: String
                if skippedCount > 0 {
                    message = "Rebuilt \(completedCount) of \(totalCount) embeddings · \(skippedCount) skipped"
                } else {
                    message = "Rebuilt \(completedCount) of \(totalCount) embeddings"
                }
                await MainActor.run {
                    self.replacePrivateAdaptationProgress(
                        PrivateAdaptationProgress(
                            step: .reindexLibrary,
                            message: message,
                            stepProgress: completed,
                            overallProgress: (2.0 + completed) / 3.0,
                            elapsedSeconds: elapsedSeconds,
                            remainingSeconds: remainingSeconds
                        )
                    )
                }
            }

            if skippedCount == missingEmbeddings.count {
                await MainActor.run {
                    self.replacePrivateAdaptationProgress(
                        PrivateAdaptationProgress(
                            step: .reindexLibrary,
                            message: "No searchable images could be rebuilt",
                            stepProgress: 1.0,
                            overallProgress: 1.0,
                            elapsedSeconds: Int(Date().timeIntervalSince(startDate)),
                            remainingSeconds: 0
                        )
                    )
                }
            }
        }

        folderSearchIndex = try searchCoordinator(database: database).folderSearchIndex(folderAbsolutePath: indexSummary.folderPath)
        applyLibrarySearchState(try folderSearchState(for: indexSummary.folderPath, database: database))
        try await refreshVisibleResults(for: url)
    }
}
