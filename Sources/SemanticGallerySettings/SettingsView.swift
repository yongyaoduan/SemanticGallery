import SwiftUI
import SemanticGalleryCore
import SemanticGalleryUI

public struct SettingsView: View {
    @Bindable private var statusStore: AppStatusStore
    @Bindable private var libraryStateStore: LibraryStateStore
    private let chooseFolder: () -> Void
    private let startPrivateAdaptation: () -> Void

    public init(
        statusStore: AppStatusStore,
        libraryStateStore: LibraryStateStore,
        chooseFolder: @escaping () -> Void,
        startPrivateAdaptation: @escaping () -> Void
    ) {
        self.statusStore = statusStore
        self.libraryStateStore = libraryStateStore
        self.chooseFolder = chooseFolder
        self.startPrivateAdaptation = startPrivateAdaptation
    }

    public var body: some View {
        ScrollView {
            settingsContent
        }
        .frame(minWidth: 720, minHeight: 420)
        .background(MuseumPaperTheme.backgroundTop)
        .alert(
            privateAdaptationAlertTitle,
            isPresented: privateAdaptationAlertBinding,
            actions: {
                Button("OK", role: .cancel) {
                    libraryStateStore.privateAdaptationNotice = nil
                }
            },
            message: {
                Text(libraryStateStore.privateAdaptationNotice ?? "")
            }
        )
    }

    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            SettingsGroupView(title: "Status") {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    let presentation = statusPresentation(at: context.date)
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(alignment: .center, spacing: 14) {
                            Text(presentation.title)
                                .font(.system(size: 28, weight: .semibold, design: .serif))
                                .foregroundStyle(MuseumPaperTheme.ink)
                                .accessibilityIdentifier("settings-status-title")

                            Spacer(minLength: 0)

                            if let activity = presentation.activity {
                                HStack(alignment: .center, spacing: 12) {
                                    ProgressView()
                                        .controlSize(.small)
                                        .tint(MuseumPaperTheme.accentStrong)
                                        .accessibilityIdentifier("settings-status-spinner")

                                    Text(activity.percentageText)
                                        .accessibilityIdentifier("settings-status-progress")

                                    Text(activity.elapsedText)
                                        .accessibilityIdentifier("settings-status-elapsed")

                                    Text(activity.remainingText)
                                        .accessibilityIdentifier("settings-status-remaining")

                                    Text(activity.estimatedCompletionText)
                                        .accessibilityIdentifier("settings-status-estimated-end")
                                }
                                .font(.system(size: 13, weight: .semibold))
                                .monospacedDigit()
                                .foregroundStyle(MuseumPaperTheme.accentStrong)
                                .frame(maxWidth: .infinity, alignment: .trailing)
                                .accessibilityElement(children: .contain)
                                .accessibilityIdentifier("settings-status-activity")
                            }
                        }

                        if let message = presentation.message {
                            Text(message)
                                .font(.system(size: 13))
                                .foregroundStyle(MuseumPaperTheme.mutedInk)
                                .fixedSize(horizontal: false, vertical: true)
                                .accessibilityIdentifier("settings-status-message")
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            SettingsGroupView(title: "Library Folder") {
                VStack(alignment: .leading, spacing: 10) {
                    if let selectedFolder = libraryStateStore.selectedFolder {
                        Text(selectedFolder.path(percentEncoded: false))
                            .foregroundStyle(MuseumPaperTheme.ink)
                            .accessibilityIdentifier("settings-selected-folder")
                    } else {
                        Text("Choose the folder you want to open as a searchable archive.")
                            .foregroundStyle(MuseumPaperTheme.mutedInk)
                            .accessibilityIdentifier("settings-library-empty-prompt")
                    }

                    Button("Choose Folder", action: chooseFolder)
                        .buttonStyle(.borderedProminent)
                        .tint(MuseumPaperTheme.accent)
                        .disabled(isBusy)
                        .accessibilityIdentifier("settings-choose-folder-button")
                }
            }

            SettingsGroupView(title: "Private Album Adaptation") {
                VStack(alignment: .leading, spacing: 12) {
                    Text("When the library is ready, adaptation will prepare data, train locally, and rebuild the search index.")
                        .font(.system(size: 14))
                        .foregroundStyle(MuseumPaperTheme.mutedInk)

                    Button("Start Private Adaptation", action: startPrivateAdaptation)
                        .buttonStyle(.borderedProminent)
                        .tint(MuseumPaperTheme.accent)
                        .disabled(libraryStateStore.selectedFolder == nil || isBusy)
                        .accessibilityIdentifier("start-private-adaptation-button")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("private-adaptation-card")
            }
        }
        .padding(28)
    }

    private func statusPresentation(at now: Date) -> SettingsStatusPresentation {
        switch statusStore.status {
        case .indexing:
            return SettingsStatusPresentation(
                title: StatusTitleFormatter.title(
                    for: statusStore.status,
                    selectedFolderExists: libraryStateStore.selectedFolder != nil,
                    supportedImageCount: libraryStateStore.supportedImageCount,
                    searchableImageCount: libraryStateStore.searchableImageCount
                ),
                message: libraryStateStore.searchIssue,
                activity: activityPresentation(
                    snapshot: StatusActivityMeter.folderPreparation(libraryStateStore.folderPreparationProgress, now: now),
                    now: now
                )
            )
        case .training:
            return SettingsStatusPresentation(
                title: StatusTitleFormatter.title(
                    for: statusStore.status,
                    selectedFolderExists: libraryStateStore.selectedFolder != nil,
                    supportedImageCount: libraryStateStore.supportedImageCount,
                    searchableImageCount: libraryStateStore.searchableImageCount
                ),
                message: libraryStateStore.searchIssue,
                activity: activityPresentation(
                    snapshot: StatusActivityMeter.privateAdaptation(libraryStateStore.privateAdaptationProgress, now: now),
                    now: now
                )
            )
        case .searching, .readyWithoutFolder, .installRequired, .ready, .installing, .installFailed, .installComplete, .uninstalling:
            return SettingsStatusPresentation(
                title: StatusTitleFormatter.title(
                    for: statusStore.status,
                    selectedFolderExists: libraryStateStore.selectedFolder != nil,
                    supportedImageCount: libraryStateStore.supportedImageCount,
                    searchableImageCount: libraryStateStore.searchableImageCount
                ),
                message: libraryStateStore.searchIssue
            )
        }
    }

    private var isBusy: Bool {
        switch statusStore.status {
        case .indexing, .training:
            return true
        case .installRequired, .installing, .installFailed, .installComplete, .readyWithoutFolder, .searching, .ready, .uninstalling:
            return false
        }
    }

    private var privateAdaptationAlertBinding: Binding<Bool> {
        Binding(
            get: { libraryStateStore.privateAdaptationNotice != nil },
            set: { isPresented in
                if isPresented == false {
                    libraryStateStore.privateAdaptationNotice = nil
                }
            }
        )
    }

    private var privateAdaptationAlertTitle: String {
        if let supportedImageCount = libraryStateStore.supportedImageCount,
           supportedImageCount < 100 {
            return "At least 100 images are needed"
        }
        return "Private Album Adaptation"
    }

    private func activityPresentation(
        snapshot: StatusActivitySnapshot?,
        now: Date
    ) -> SettingsStatusActivity? {
        guard let snapshot else {
            return nil
        }

        return SettingsStatusActivity(
            percentageText: "\(Int((snapshot.progress * 100).rounded()))%",
            elapsedText: "Elapsed \(durationText(for: snapshot.elapsedSeconds))",
            remainingText: "Left \(remainingDurationText(for: snapshot.remainingSeconds))",
            estimatedCompletionText: "Ends \(estimatedCompletionText(for: snapshot.estimatedCompletionDate, now: now))"
        )
    }

    private func durationText(for totalSeconds: Int) -> String {
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60

        if hours > 0 {
            return "\(hours)h \(minutes)m"
        }
        if minutes > 0 {
            return "\(minutes)m \(seconds)s"
        }
        return "\(seconds)s"
    }

    private func remainingDurationText(for remainingSeconds: Int?) -> String {
        guard let remainingSeconds else {
            return "--"
        }
        return durationText(for: remainingSeconds)
    }

    private func estimatedCompletionText(for date: Date?, now: Date) -> String {
        guard let date else {
            return "--"
        }

        if Calendar.current.isDate(date, inSameDayAs: now) {
            return date.formatted(date: .omitted, time: .shortened)
        }
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct SettingsStatusPresentation {
    let title: String
    var message: String? = nil
    var activity: SettingsStatusActivity? = nil
}

private struct SettingsStatusActivity {
    let percentageText: String
    let elapsedText: String
    let remainingText: String
    let estimatedCompletionText: String
}
