import SwiftUI
import SemanticGallerySearch
import SemanticGallerySettings

public struct SemanticGalleryRootView: View {
    @Bindable private var controller: SemanticGalleryController

    public init(controller: SemanticGalleryController) {
        self.controller = controller
    }

    public var body: some View {
        Group {
            switch controller.statusStore.status {
            case .readyWithoutFolder, .indexing, .searching, .training, .ready, .installRequired, .installing, .installFailed, .installComplete, .uninstalling:
                UsageView(
                    libraryStateStore: controller.libraryStateStore,
                    workspaceStateStore: controller.workspaceStateStore,
                    thumbnailStore: controller.thumbnailStore,
                    runSearch: controller.runSearch,
                    enterSelectionMode: controller.enterSelectionMode,
                    leaveSelectionMode: controller.leaveSelectionMode,
                    toggleSelection: controller.toggleSelection,
                    selectAll: controller.selectAllVisibleResults,
                    clearSelection: controller.clearSelection,
                    deleteSelection: controller.deleteSelection,
                    openPreview: controller.openPreview,
                    closePreview: controller.closePreview,
                    togglePreviewMetadata: controller.togglePreviewMetadata,
                    searchSimilarToPreview: controller.searchSimilarToPreviewItem,
                    deletePreview: controller.deletePreviewItem,
                    showNextPreviewItem: controller.showNextPreviewItem,
                    showPreviousPreviewItem: controller.showPreviousPreviewItem
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

public struct SemanticGallerySettingsScene: View {
    @Bindable private var controller: SemanticGalleryController

    public init(controller: SemanticGalleryController) {
        self.controller = controller
    }

    public var body: some View {
        SettingsView(
            statusStore: controller.statusStore,
            libraryStateStore: controller.libraryStateStore,
            chooseFolder: controller.chooseFolder,
            startPrivateAdaptation: controller.startPrivateAdaptation
        )
    }
}
