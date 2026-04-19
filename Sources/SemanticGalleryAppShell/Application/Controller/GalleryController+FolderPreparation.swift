import Foundation
import SemanticGalleryCore
import SemanticGalleryIndex
import SemanticGalleryML
import SemanticGalleryPersistence

struct FolderPreparationOutcome: Sendable {
    let folderPath: String
    let supportedImageCount: Int
    let reusedEmbeddingCount: Int
    let generatedEmbeddingCount: Int
    let skippedImageCount: Int
    let searchState: FolderSearchState
}

struct CachedVectorQuery {
    let encoderVersion: String
    let source: VectorQuerySource
    let vector: [Double]
}

enum VectorQuerySource: Equatable {
    case text(String)
    case image(Data)
}

func searchIssueMessage(for error: Error) -> String {
    switch error {
    case GalleryEmbeddingError.invalidModelArtifact:
        return "Semantic search model files could not be loaded from the app bundle."
    case GalleryEmbeddingError.invalidStage1Weights:
        return "Semantic search weights could not be loaded from the app bundle."
    case GalleryEmbeddingError.invalidImageData:
        return "Semantic search could not prepare embeddings for this folder."
    case GalleryEmbeddingError.invalidAdapterArtifact:
        return "Semantic search could not load the folder adaptation, so only the published encoder is available."
    default:
        return "Semantic search could not finish preparing embeddings for this folder."
    }
}

func folderPreparationMetadata(
    for url: URL,
    outcome: FolderPreparationOutcome,
    encoderVersion: String
) -> [String: String] {
    [
        "encoder_version": encoderVersion,
        "folder": url.path(percentEncoded: false),
        "generated_embeddings": "\(outcome.generatedEmbeddingCount)",
        "reused_embeddings": "\(outcome.reusedEmbeddingCount)",
        "searchable_images": "\(outcome.searchState.searchableImageCount)",
        "skipped_images": "\(outcome.skippedImageCount)",
        "supported_images": "\(outcome.supportedImageCount)",
    ]
}

func prepareFolderSelection(
    at url: URL,
    database: LibraryDatabase,
    embeddingService: any GalleryEmbeddingService,
    encoderVersion: String,
    stepDelayNanoseconds: UInt64,
    onProgress: @escaping @Sendable (FolderPreparationProgress) async -> Void
) async throws -> FolderPreparationOutcome {
    let totalSteps = 6.0

    let accessStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .requestFolderAccess,
            message: "Secure access is ready",
            stepProgress: 1.0,
            overallProgress: 1.0 / totalSteps,
            elapsedSeconds: Int(Date().timeIntervalSince(accessStart)),
            remainingSeconds: 0
        )
    )
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let scanStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Scanning supported images",
            stepProgress: 0,
            overallProgress: 1.0 / totalSteps,
            elapsedSeconds: 0,
            remainingSeconds: nil,
            recordedAt: Date()
        )
    )
    let indexSummary = try await FolderIndexer(database: database).rebuildIndex(for: url) { snapshot in
        let stepProgress: Double
        if snapshot.totalCount == 0 {
            stepProgress = 1.0
        } else {
            stepProgress = Double(snapshot.processedCount) / Double(snapshot.totalCount)
        }
        let elapsed = Int(Date().timeIntervalSince(scanStart))
        let remaining = snapshot.processedCount > 0 && snapshot.totalCount > snapshot.processedCount
            ? Int((Double(elapsed) / Double(snapshot.processedCount)) * Double(snapshot.totalCount - snapshot.processedCount))
            : 0

        await onProgress(
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Scanned \(snapshot.processedCount) of \(snapshot.totalCount) images · \(snapshot.uniqueAssetCount) unique assets",
                stepProgress: stepProgress,
                overallProgress: (1.0 + stepProgress) / totalSteps,
                elapsedSeconds: elapsed,
                remainingSeconds: remaining,
                recordedAt: Date()
            )
        )
    }

    if indexSummary.fileCount == 0 {
        await onProgress(
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Scanned 0 of 0 images",
                stepProgress: 1.0,
                overallProgress: 2.0 / totalSteps,
                elapsedSeconds: Int(Date().timeIntervalSince(scanStart)),
                remainingSeconds: 0,
                recordedAt: Date()
            )
        )
    } else {
        await onProgress(
            FolderPreparationProgress(
                step: .scanSupportedImages,
                message: "Scanned \(indexSummary.fileCount) of \(indexSummary.fileCount) images · \(indexSummary.uniqueAssetCount) unique assets",
                stepProgress: 1.0,
                overallProgress: 2.0 / totalSteps,
                elapsedSeconds: Int(Date().timeIntervalSince(scanStart)),
                remainingSeconds: 0,
                recordedAt: Date()
            )
        )
    }
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let missingEmbeddings = try database.missingEmbeddingWork(
        inFolderAbsolutePath: indexSummary.folderPath,
        encoderVersion: encoderVersion
    )
    let reusedCount = max(0, indexSummary.uniqueAssetCount - missingEmbeddings.count)
    let reuseStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .reuseExistingEmbeddings,
            message: "Reused \(reusedCount) embeddings, \(missingEmbeddings.count) still needed",
            stepProgress: 1.0,
            overallProgress: 3.0 / totalSteps,
            elapsedSeconds: Int(Date().timeIntervalSince(reuseStart)),
            remainingSeconds: 0,
            recordedAt: Date()
        )
    )
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let embeddingStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .generateEmbeddings,
            message: missingEmbeddings.isEmpty ? "All embeddings are already available" : "Generating search embeddings",
            stepProgress: 0,
            overallProgress: 3.0 / totalSteps,
            elapsedSeconds: 0,
            remainingSeconds: nil,
            recordedAt: Date()
        )
    )
    var skippedImageCount = 0
    if missingEmbeddings.isEmpty {
        await onProgress(
            FolderPreparationProgress(
                step: .generateEmbeddings,
                message: "Generated 0 of 0 embeddings",
                stepProgress: 1.0,
                overallProgress: 4.0 / totalSteps,
                elapsedSeconds: Int(Date().timeIntervalSince(embeddingStart)),
                remainingSeconds: 0,
                recordedAt: Date()
            )
        )
    } else {
        skippedImageCount = try await storeEmbeddingsInBatches(
            missingEmbeddings,
            database: database,
            embeddingService: embeddingService,
            encoderVersion: encoderVersion
        ) { completedCount, totalCount, skippedCount, elapsedSeconds, remainingSeconds in
            let completed = totalCount == 0 ? 1.0 : Double(completedCount) / Double(totalCount)
            let statusText = skippedCount > 0
                ? "Prepared \(completedCount) of \(totalCount) images · \(skippedCount) skipped"
                : "Prepared \(completedCount) of \(totalCount) embeddings"
            await onProgress(
                FolderPreparationProgress(
                    step: .generateEmbeddings,
                    message: statusText,
                    stepProgress: completed,
                    overallProgress: (3.0 + completed) / totalSteps,
                    elapsedSeconds: elapsedSeconds,
                    remainingSeconds: remainingSeconds,
                    recordedAt: Date()
                )
            )
        }
    }
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let supportedCount = try database.visibleFileInstances(
        inFolderAbsolutePath: indexSummary.folderPath,
        limit: Int.max
    ).count
    let searchableCount = try database.embeddedFileCount(
        inFolderAbsolutePath: indexSummary.folderPath,
        encoderVersion: encoderVersion
    )
    let buildStart = Date()
    let buildMessage: String
    if searchableCount < supportedCount {
        let noun = supportedCount == 1 ? "image" : "images"
        buildMessage = "\(searchableCount) of \(supportedCount) \(noun) ready for search"
    } else {
        let noun = searchableCount == 1 ? "image" : "images"
        buildMessage = "\(searchableCount) \(noun) ready for search"
    }
    await onProgress(
        FolderPreparationProgress(
            step: .buildSearchView,
            message: buildMessage,
            stepProgress: 1.0,
            overallProgress: 5.0 / totalSteps,
            elapsedSeconds: Int(Date().timeIntervalSince(buildStart)),
            remainingSeconds: 0,
            recordedAt: Date()
        )
    )
    try await pauseBetweenFolderSteps(stepDelayNanoseconds)

    let finalizeStart = Date()
    await onProgress(
        FolderPreparationProgress(
            step: .finalizeFolderSelection,
            message: "Folder is ready",
            stepProgress: 1.0,
            overallProgress: 1.0,
            elapsedSeconds: Int(Date().timeIntervalSince(finalizeStart)),
            remainingSeconds: 0,
            recordedAt: Date()
        )
    )

    return FolderPreparationOutcome(
        folderPath: indexSummary.folderPath,
        supportedImageCount: supportedCount,
        reusedEmbeddingCount: reusedCount,
        generatedEmbeddingCount: Swift.max(0, searchableCount - reusedCount),
        skippedImageCount: skippedImageCount,
        searchState: FolderSearchState(
            supportedImageCount: supportedCount,
            searchableImageCount: searchableCount
        )
    )
}

func pauseBetweenFolderSteps(_ stepDelayNanoseconds: UInt64) async throws {
    if stepDelayNanoseconds > 0 {
        try await Task.sleep(nanoseconds: stepDelayNanoseconds)
    } else {
        await Task.yield()
    }
}

func storeEmbeddingsInBatches(
    _ workItems: [EmbeddingWorkRecord],
    database: LibraryDatabase,
    embeddingService: any GalleryEmbeddingService,
    encoderVersion: String,
    onProgress: @escaping @Sendable (Int, Int, Int, Int, Int?) async -> Void
) async throws -> Int {
    guard workItems.isEmpty == false else {
        await onProgress(0, 0, 0, 0, 0)
        return 0
    }

    let startDate = Date()
    let planner = StableBatchPlanner(batchSize: 12)
    let batches = planner.makeBatches(workItems) {
        "\($0.assetID):\($0.representativeAbsolutePath)"
    }

    var completedCount = 0
    var skippedCount = 0
    for batch in batches {
        do {
            let vectors = try await embeddingService.encodeImages(
                at: batch.map { URL(filePath: $0.representativeAbsolutePath) }
            )
            for (item, vector) in zip(batch, vectors) {
                try database.upsertEmbedding(
                    assetID: item.assetID,
                    encoderVersion: encoderVersion,
                    vector: vector
                )
            }
        } catch {
            for item in batch {
                do {
                    let vector = try await embeddingService.encodeImage(
                        at: URL(filePath: item.representativeAbsolutePath)
                    )
                    try database.upsertEmbedding(
                        assetID: item.assetID,
                        encoderVersion: encoderVersion,
                        vector: vector
                    )
                } catch let galleryError as GalleryEmbeddingError where galleryError == .invalidImageData {
                    skippedCount += 1
                } catch {
                    throw error
                }
            }
        }

        completedCount += batch.count
        let elapsedSeconds = Int(Date().timeIntervalSince(startDate))
        let remainingCount = max(0, workItems.count - completedCount)
        let remainingSeconds: Int?
        if completedCount > 0 {
            remainingSeconds = Int((Double(elapsedSeconds) / Double(completedCount)) * Double(remainingCount))
        } else {
            remainingSeconds = nil
        }
        await onProgress(
            completedCount,
            workItems.count,
            skippedCount,
            elapsedSeconds,
            remainingSeconds
        )
    }

    return skippedCount
}
