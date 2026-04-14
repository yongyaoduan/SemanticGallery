import SwiftUI
import SemanticGalleryCore
import SemanticGallerySettings

@main
struct SemanticGalleryApp: App {
    @State private var container = AppContainer()
    private let folderPickerCoordinator = FolderPickerCoordinator()
    private let folderPreparationCoordinator = FolderPreparationCoordinator()

    var body: some Scene {
        WindowGroup {
            RootContentView(
                statusStore: container.statusStore,
                libraryStateStore: container.libraryStateStore,
                chooseFolder: chooseFolder
            )
                .frame(minWidth: 1200, minHeight: 780)
        }
        Settings {
            SettingsView(
                statusStore: container.statusStore,
                libraryStateStore: container.libraryStateStore,
                chooseFolder: chooseFolder,
                startUninstall: {}
            )
        }
    }

    @MainActor
    private func chooseFolder() {
        guard let url = folderPickerCoordinator.pickFolder() else {
            return
        }

        container.statusStore.status = .preparingFolder
        container.libraryStateStore.folderPreparationProgress = []

        Task {
            let progress = try await folderPreparationCoordinator.prepareFolder(at: url) { item in
                await MainActor.run {
                    container.libraryStateStore.folderPreparationProgress.append(item)
                }
            }

            await MainActor.run {
                container.libraryStateStore.selectedFolder = url
                container.libraryStateStore.folderPreparationProgress = progress
                container.statusStore.status = .ready
                container.libraryStateStore.folderPreparationProgress = []
            }
        }
    }
}
