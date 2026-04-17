import Foundation
import Testing
@testable import SemanticGalleryCore

@Test
@MainActor
func initialAppStatusStartsInReadyWithoutFolder() {
    let store = AppStatusStore()
    #expect(store.status == .readyWithoutFolder)
}

@Test
func folderPreparationProgressUsesWeightedStepProgress() {
    let progress = StatusProgressMeter.folderPreparation([
        FolderPreparationProgress(
            step: .requestFolderAccess,
            message: "Ready",
            stepProgress: 1.0,
            overallProgress: 1.0 / 6.0
        ),
        FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Scanning",
            stepProgress: 0.5,
            overallProgress: 0.25
        ),
    ])

    #expect(abs(progress - ((1.0 / 6.0) + (0.5 / 6.0))) < 0.000_1)
}

@Test
func privateAdaptationProgressUsesWeightedStepProgress() {
    let progress = StatusProgressMeter.privateAdaptation([
        PrivateAdaptationProgress(
            step: .prepareData,
            message: "Prepared",
            stepProgress: 1.0,
            overallProgress: 1.0 / 3.0
        ),
        PrivateAdaptationProgress(
            step: .trainModel,
            message: "Training",
            stepProgress: 0.25,
            overallProgress: (1.0 / 3.0) + (0.25 / 3.0)
        ),
    ])

    #expect(abs(progress - ((1.0 / 3.0) + (0.25 / 3.0))) < 0.000_1)
}

@Test
func weightedProgressUsesLatestSnapshotPerStepAndClampsToBounds() {
    let progress = StatusProgressMeter.folderPreparation([
        FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Early",
            stepProgress: 0.1,
            overallProgress: 0.2
        ),
        FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Latest",
            stepProgress: 1.6,
            overallProgress: 0.4
        ),
        FolderPreparationProgress(
            step: .reuseExistingEmbeddings,
            message: "Negative",
            stepProgress: -1.0,
            overallProgress: 0.4
        ),
    ])

    #expect(abs(progress - (1.0 / 6.0)) < 0.000_1)
}

@Test
func folderPreparationActivityUsesWeightedProgressAndLiveTiming() {
    let now = Date(timeIntervalSince1970: 2_000)
    let activity = StatusActivityMeter.folderPreparation([
        FolderPreparationProgress(
            step: .requestFolderAccess,
            message: "Authorized",
            stepProgress: 1.0,
            overallProgress: 1.0 / 6.0,
            elapsedSeconds: 0,
            remainingSeconds: 0,
            recordedAt: now.addingTimeInterval(-20)
        ),
        FolderPreparationProgress(
            step: .scanSupportedImages,
            message: "Scanning",
            stepProgress: 0.5,
            overallProgress: 0.25,
            elapsedSeconds: 10,
            remainingSeconds: 20,
            recordedAt: now.addingTimeInterval(-5)
        ),
    ], now: now)

    #expect(activity != nil)
    #expect(abs((activity?.progress ?? 0) - 0.25) < 0.000_1)
    #expect(activity?.elapsedSeconds == 15)
    #expect(activity?.remainingSeconds == 15)
    #expect(activity?.estimatedCompletionDate == now.addingTimeInterval(15))
}

@Test
func privateAdaptationActivityUsesMostRecentSnapshot() {
    let now = Date(timeIntervalSince1970: 5_000)
    let expectedProgress = (1.0 / 3.0) + (0.4 / 3.0)
    let activity = StatusActivityMeter.privateAdaptation([
        PrivateAdaptationProgress(
            step: .prepareData,
            message: "Preparing",
            stepProgress: 1.0,
            overallProgress: 1.0 / 3.0,
            elapsedSeconds: 12,
            remainingSeconds: 0,
            recordedAt: now.addingTimeInterval(-30)
        ),
        PrivateAdaptationProgress(
            step: .trainModel,
            message: "Training",
            stepProgress: 0.4,
            overallProgress: (1.0 / 3.0) + (0.4 / 3.0),
            elapsedSeconds: 40,
            remainingSeconds: 60,
            recordedAt: now.addingTimeInterval(-10)
        ),
    ], now: now)

    #expect(activity != nil)
    #expect(abs((activity?.progress ?? 0) - expectedProgress) < 0.000_1)
    #expect(activity?.elapsedSeconds == 50)
    #expect(activity?.remainingSeconds == 50)
    #expect(activity?.estimatedCompletionDate == now.addingTimeInterval(50))
}

@Test
func statusActivityReturnsNilWhenNoProgressExists() {
    #expect(StatusActivityMeter.folderPreparation([]) == nil)
    #expect(StatusActivityMeter.privateAdaptation([]) == nil)
}

@Test
func statusTitleFormatterUsesSearchableImageCountsWhenAvailable() {
    #expect(
        StatusTitleFormatter.title(
            for: .readyWithoutFolder,
            selectedFolderExists: false,
            supportedImageCount: nil,
            searchableImageCount: nil
        ) == "No folder selected"
    )
    #expect(
        StatusTitleFormatter.title(
            for: .ready,
            selectedFolderExists: true,
            supportedImageCount: 1,
            searchableImageCount: 1
        ) == "1 image ready"
    )
    #expect(
        StatusTitleFormatter.title(
            for: .ready,
            selectedFolderExists: true,
            supportedImageCount: 24,
            searchableImageCount: 24
        ) == "24 images ready"
    )
    #expect(
        StatusTitleFormatter.title(
            for: .ready,
            selectedFolderExists: true,
            supportedImageCount: 24,
            searchableImageCount: 18
        ) == "18 of 24 images ready"
    )
    #expect(
        StatusTitleFormatter.title(
            for: .ready,
            selectedFolderExists: true,
            supportedImageCount: 1,
            searchableImageCount: 0
        ) == "0 of 1 image ready"
    )
    #expect(
        StatusTitleFormatter.title(
            for: .indexing,
            selectedFolderExists: true,
            supportedImageCount: 24,
            searchableImageCount: 18
        ) == "Indexing"
    )
    #expect(
        StatusTitleFormatter.title(
            for: .training,
            selectedFolderExists: true,
            supportedImageCount: 24,
            searchableImageCount: 18
        ) == "Adapting"
    )
}

@Test
func folderSyncPolicyUsesTenPercentThresholdRoundedDown() {
    #expect(FolderSyncPolicy.matrixRefreshThreshold(forVisibleImageCount: 0) == 0)
    #expect(FolderSyncPolicy.matrixRefreshThreshold(forVisibleImageCount: 8) == 0)
    #expect(FolderSyncPolicy.matrixRefreshThreshold(forVisibleImageCount: 10) == 1)
    #expect(FolderSyncPolicy.matrixRefreshThreshold(forVisibleImageCount: 29) == 2)
}

@Test
func folderSyncPolicyRefreshesMissingViewsOrEnoughAccumulatedChanges() {
    #expect(FolderSyncPolicy.shouldRebuildActiveView(currentViewExists: false, pendingChangeCount: 1, visibleImageCount: 50))
    #expect(FolderSyncPolicy.shouldRebuildActiveView(currentViewExists: true, pendingChangeCount: 4, visibleImageCount: 50) == false)
    #expect(FolderSyncPolicy.shouldRebuildActiveView(currentViewExists: true, pendingChangeCount: 5, visibleImageCount: 50))
}
