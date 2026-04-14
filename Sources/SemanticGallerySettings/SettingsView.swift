import SwiftUI
import SemanticGalleryCore
import SemanticGalleryUI

public struct SettingsView: View {
    @Bindable private var statusStore: AppStatusStore
    @Bindable private var libraryStateStore: LibraryStateStore
    private let chooseFolder: () -> Void
    private let startUninstall: () -> Void
    @State private var showUninstallConfirmation = false

    public init(
        statusStore: AppStatusStore,
        libraryStateStore: LibraryStateStore,
        chooseFolder: @escaping () -> Void,
        startUninstall: @escaping () -> Void
    ) {
        self.statusStore = statusStore
        self.libraryStateStore = libraryStateStore
        self.chooseFolder = chooseFolder
        self.startUninstall = startUninstall
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                SettingsGroupView(title: "Status") {
                    Text(statusTitle)
                        .font(.system(size: 24, weight: .semibold, design: .serif))
                        .foregroundStyle(MuseumPaperTheme.ink)
                }

                SettingsGroupView(title: "Library Folder") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(libraryStateStore.selectedFolder?.path(percentEncoded: false) ?? "No folder selected")
                            .foregroundStyle(MuseumPaperTheme.ink)

                        Button("Choose Folder", action: chooseFolder)
                            .buttonStyle(.borderedProminent)
                            .tint(MuseumPaperTheme.accent)
                            .accessibilityIdentifier("choose-folder-button")
                    }

                    if libraryStateStore.folderPreparationProgress.isEmpty == false {
                        VStack(alignment: .leading, spacing: 12) {
                            LayeredProgressBar(progress: overallProgress)
                                .frame(height: 12)

                            ForEach(Array(libraryStateStore.folderPreparationProgress.enumerated()), id: \.offset) { _, item in
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text(title(for: item.step))
                                            .foregroundStyle(MuseumPaperTheme.ink)
                                        Spacer()
                                        Text(item.message)
                                            .foregroundStyle(MuseumPaperTheme.mutedInk)
                                    }
                                    LayeredProgressBar(progress: item.progress)
                                        .frame(height: 10)
                                }
                            }
                        }
                        .accessibilityIdentifier("folder-preparation-progress-group")
                    }
                }

                SettingsGroupView(title: "Private Album Adaptation") {
                    Text("Private adaptation stays in Settings. It requires at least 100 supported images and arrives after the indexing stage is in place.")
                        .foregroundStyle(MuseumPaperTheme.mutedInk)
                    Button("Start Adaptation") {}
                        .disabled(true)
                }

                SettingsGroupView(title: "Uninstall") {
                    Button("Uninstall SemanticGallery", role: .destructive) {
                        showUninstallConfirmation = true
                    }
                        .accessibilityIdentifier("uninstall-button")
                }
            }
            .padding(28)
        }
        .frame(minWidth: 720, minHeight: 560)
        .background(MuseumPaperTheme.backgroundTop)
        .confirmationDialog(
            "Uninstall SemanticGallery?",
            isPresented: $showUninstallConfirmation,
            actions: {
                Button("Remove App Data", role: .destructive, action: startUninstall)
                Button("Cancel", role: .cancel) {}
            },
            message: {
                Text("This removes app data, downloaded models, caches, and bookmarks, but it keeps your photo folders.")
            }
        )
    }

    private var overallProgress: Double {
        libraryStateStore.folderPreparationProgress.last?.progress ?? 0
    }

    private var statusTitle: String {
        switch statusStore.status {
        case .installRequired:
            return "Needs Installation"
        case .installing:
            return "Installing"
        case .installFailed:
            return "Install Failed"
        case .installComplete:
            return "Installation Complete"
        case .readyWithoutFolder:
            return "Idle"
        case .preparingFolder:
            return "Indexing"
        case .ready:
            return "Idle"
        case .uninstalling:
            return "Uninstalling"
        }
    }

    private func title(for step: FolderPreparationStep) -> String {
        switch step {
        case .requestFolderAccess:
            return "Authorize"
        case .scanSupportedImages:
            return "Scan Images"
        case .persistBookmark:
            return "Save Access"
        case .finalizeFolderSelection:
            return "Finalize"
        }
    }
}
