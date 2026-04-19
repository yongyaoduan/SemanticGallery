import AppKit
import SwiftUI
import SemanticGallerySearch
import SemanticGallerySettings

public struct AppRootView: View {
    @Bindable private var controller: GalleryController

    public init(controller: GalleryController) {
        self.controller = controller
    }

    public var body: some View {
        Group {
            switch controller.statusState.status {
            case .readyWithoutFolder, .indexing, .searching, .training, .ready, .installRequired, .installing, .installFailed, .installComplete, .uninstalling:
                UsageView(
                    libraryState: controller.libraryState,
                    usageState: controller.usageState,
                    thumbnailCache: controller.thumbnailCache,
                    runSearch: controller.runSearch,
                    enterSelectionMode: controller.enterSelectionMode,
                    leaveSelectionMode: controller.leaveSelectionMode,
                    toggleSelection: controller.toggleSelection,
                    selectAll: controller.selectAllVisibleResults,
                    clearSelection: controller.clearSelection,
                    deleteSelection: controller.deleteSelectedResults,
                    openPreview: controller.openPreview,
                    closePreview: controller.closePreview,
                    togglePreviewMetadata: controller.togglePreviewMetadata,
                    searchSimilarToPreview: controller.searchSimilarToPreviewItem,
                    deletePreview: controller.deletePreviewResult,
                    showNextPreviewItem: controller.showNextPreviewItem,
                    showPreviousPreviewItem: controller.showPreviousPreviewItem
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            promoteWindowToFront()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                promoteWindowToFront()
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                promoteWindowToFront()
            }
        }
    }

    private func promoteWindowToFront() {
        let application = NSApplication.shared
        application.unhide(nil)
        application.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [.activateAllWindows])
        if let window = application.keyWindow
            ?? application.mainWindow
            ?? application.windows.first(where: { $0.isVisible })
            ?? application.windows.first {
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
        }
    }
}

public struct AppSettingsScene: View {
    @Bindable private var controller: GalleryController

    public init(controller: GalleryController) {
        self.controller = controller
    }

    public var body: some View {
        SettingsView(
            statusState: controller.statusState,
            libraryState: controller.libraryState,
            chooseFolder: controller.chooseFolder,
            startPrivateAdaptation: controller.startPrivateAdaptation
        )
    }
}
