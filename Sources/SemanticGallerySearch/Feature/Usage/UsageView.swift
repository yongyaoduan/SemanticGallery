import AppKit
import SwiftUI
import SemanticGalleryCore
import SemanticGalleryUI

public struct UsageView: View {
    @Environment(\.openSettings) private var openSettings
    @Bindable private var libraryState: LibraryState
    @Bindable private var usageState: UsageState

    private let thumbnailCache: ThumbnailCache
    private let runSearch: () -> Void
    private let enterSelectionMode: () -> Void
    private let leaveSelectionMode: () -> Void
    private let toggleSelection: (Int64) -> Void
    private let selectAll: () -> Void
    private let clearSelection: () -> Void
    private let deleteSelection: () -> Void
    private let openPreview: (Int64) -> Void
    private let closePreview: () -> Void
    private let togglePreviewMetadata: () -> Void
    private let searchSimilarToPreview: () -> Void
    private let deletePreview: () -> Void
    private let showNextPreviewItem: () -> Void
    private let showPreviousPreviewItem: () -> Void

    @State private var deleteConfirmationContext: WorkspaceDeleteContext?
    @State private var searchFocusRequest = 0

    public init(
        libraryState: LibraryState,
        usageState: UsageState,
        thumbnailCache: ThumbnailCache,
        runSearch: @escaping () -> Void,
        enterSelectionMode: @escaping () -> Void,
        leaveSelectionMode: @escaping () -> Void,
        toggleSelection: @escaping (Int64) -> Void,
        selectAll: @escaping () -> Void,
        clearSelection: @escaping () -> Void,
        deleteSelection: @escaping () -> Void,
        openPreview: @escaping (Int64) -> Void,
        closePreview: @escaping () -> Void,
        togglePreviewMetadata: @escaping () -> Void,
        searchSimilarToPreview: @escaping () -> Void,
        deletePreview: @escaping () -> Void,
        showNextPreviewItem: @escaping () -> Void,
        showPreviousPreviewItem: @escaping () -> Void
    ) {
        self.libraryState = libraryState
        self.usageState = usageState
        self.thumbnailCache = thumbnailCache
        self.runSearch = runSearch
        self.enterSelectionMode = enterSelectionMode
        self.leaveSelectionMode = leaveSelectionMode
        self.toggleSelection = toggleSelection
        self.selectAll = selectAll
        self.clearSelection = clearSelection
        self.deleteSelection = deleteSelection
        self.openPreview = openPreview
        self.closePreview = closePreview
        self.togglePreviewMetadata = togglePreviewMetadata
        self.searchSimilarToPreview = searchSimilarToPreview
        self.deletePreview = deletePreview
        self.showNextPreviewItem = showNextPreviewItem
        self.showPreviousPreviewItem = showPreviousPreviewItem
    }

    public var body: some View {
        ZStack {
            LinearGradient(
                colors: [MuseumPaperTheme.backgroundTop, MuseumPaperTheme.backgroundBottom],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            workspaceContent(for: libraryState.selectedFolder)

            if let previewItem = usageState.previewItem {
                previewOverlay(for: previewItem)
            }
        }
        .animation(.easeInOut(duration: 0.18), value: usageState.isSelectionModeEnabled)
        .animation(.easeInOut(duration: 0.18), value: usageState.previewResultID)
        .confirmationDialog(
            deleteDialogTitle,
            isPresented: deleteDialogBinding,
            actions: {
                Button("Move to Trash", role: .destructive) {
                    performConfirmedDelete()
                }
                Button("Cancel", role: .cancel) {}
            },
            message: {
                Text(deleteDialogMessage)
            }
        )
    }

    private var emptyStateWorkspaceGhost: some View {
        VStack(spacing: 14) {
            Capsule(style: .continuous)
                .fill(MuseumPaperTheme.panel.opacity(0.62))
                .frame(height: 34)
                .overlay(
                    Capsule(style: .continuous)
                        .stroke(MuseumPaperTheme.line.opacity(0.6), lineWidth: 1)
                )

            HStack(spacing: 12) {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(MuseumPaperTheme.panel.opacity(0.66))
                    .frame(height: 76)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .stroke(MuseumPaperTheme.line.opacity(0.6), lineWidth: 1)
                    )

                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(MuseumPaperTheme.panel.opacity(0.62))
                    .frame(width: 92, height: 52)
                    .overlay(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .stroke(MuseumPaperTheme.line.opacity(0.6), lineWidth: 1)
                    )

                ForEach(0..<2, id: \.self) { _ in
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(MuseumPaperTheme.panel.opacity(0.62))
                        .frame(width: 52, height: 52)
                        .overlay(
                            RoundedRectangle(cornerRadius: 22, style: .continuous)
                                .stroke(MuseumPaperTheme.line.opacity(0.6), lineWidth: 1)
                        )
                }
            }
        }
        .accessibilityHidden(true)
    }

    private func workspaceContent(for selectedFolder: URL?) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            pathBar(for: selectedFolder)
            workspaceToolbar

            if selectedFolder == nil {
                Spacer(minLength: 24)
                emptyWorkspaceGuide
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            } else if usageState.results.isEmpty {
                emptyResultsState
            } else {
                ScrollView {
                    LazyVGrid(
                        columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 5),
                        spacing: 6
                    ) {
                        ForEach(usageState.results) { item in
                            UsageResultCell(
                                item: item,
                                isSelected: usageState.isSelected(item.id),
                                showsSelection: usageState.isSelectionModeEnabled,
                                thumbnailCache: thumbnailCache
                            ) {
                                if usageState.isSelectionModeEnabled {
                                    toggleSelection(item.id)
                                } else {
                                    openPreview(item.id)
                                }
                            }
                        }
                    }
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("workspace-results-grid")
                    .padding(.bottom, 12)
                }
                .task(id: thumbnailPrefetchIDs) {
                    await thumbnailCache.prefetchImages(
                        for: Array(usageState.results.prefix(15)),
                        size: CGSize(width: 420, height: 420),
                        scale: NSScreen.main?.backingScaleFactor ?? 2
                    )
                }
            }
        }
        .padding(24)
    }

    private var workspaceToolbar: some View {
        HStack(alignment: .center, spacing: 10) {
            searchEditor
                .frame(maxWidth: .infinity)

            resultLimitControl

            selectionControls

            settingsButton
        }
    }

    private func pathBar(for selectedFolder: URL?) -> some View {
        HStack {
            Text(selectedFolder?.path(percentEncoded: false) ?? "Choose a library folder in Settings")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(selectedFolder == nil ? MuseumPaperTheme.mutedInk : MuseumPaperTheme.ink)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.horizontal, 14)
        .frame(height: WorkspaceMetrics.controlHeight)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                .fill(MuseumPaperTheme.panel.opacity(0.9))
                .overlay(
                    RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                        .stroke(MuseumPaperTheme.line, lineWidth: 1)
                )
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(selectedFolder?.path(percentEncoded: false) ?? "Choose a library folder in Settings")
        .accessibilityIdentifier("workspace-path-bar")
    }

    private var searchEditor: some View {
        activeSearchEditor
    }

    private var activeSearchEditor: some View {
        HStack(spacing: 8) {
            if let image = pastedPreviewImage {
                HStack(spacing: 8) {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 24, height: 24)
                        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))

                    Button {
                        usageState.pastedImageData = nil
                        runSearch()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(MuseumPaperTheme.mutedInk)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Remove Image")
                }
            }

            PasteAwareSearchField(
                text: $usageState.queryText,
                placeholder: "Search with words or paste an image",
                focusRequest: searchFocusRequest,
                onSubmit: runSearch,
                onImagePaste: { data in
                    usageState.queryText = ""
                    usageState.pastedImageData = data
                    runSearch()
                }
            )
            .frame(maxWidth: .infinity)

            if usageState.isSearching {
                ProgressView()
                    .controlSize(.small)
                    .tint(MuseumPaperTheme.accent)
            }

            if let searchAssistText {
                Text(searchAssistText)
                .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(usageState.isSearching ? MuseumPaperTheme.accentStrong : MuseumPaperTheme.mutedInk)
                    .accessibilityIdentifier("workspace-search-status")
            }
        }
        .padding(.horizontal, 14)
        .frame(height: WorkspaceMetrics.controlHeight)
        .background(
            RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                .fill(MuseumPaperTheme.panel)
        )
        .overlay(
            RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                .stroke(MuseumPaperTheme.line, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture {
            searchFocusRequest += 1
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("workspace-search-editor")
    }

    private var resultLimitControl: some View {
        Menu {
            ForEach([25, 50, 100, 250], id: \.self) { value in
                Button {
                    usageState.resultLimit = value
                } label: {
                    Text("\(value)")
                }
            }
        } label: {
            HStack(spacing: 7) {
                Text("\(usageState.resultLimit)")
                    .font(.system(size: 13, weight: .medium))
                    .monospacedDigit()

                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundStyle(MuseumPaperTheme.accentStrong)
            .frame(width: 82, height: WorkspaceMetrics.controlHeight)
            .background(
                RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                    .fill(MuseumPaperTheme.panel.opacity(0.92))
                    .overlay(
                        RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                            .stroke(MuseumPaperTheme.line, lineWidth: 1)
                    )
            )
        }
        .menuStyle(.borderlessButton)
        .accessibilityIdentifier("workspace-result-limit-picker")
    }

    private var selectionControls: some View {
        HStack(spacing: 10) {
            if usageState.isSelectionModeEnabled {
                selectionCountBadge

                WorkspaceIconButton(
                    systemName: isAllVisibleSelected ? "checkmark.square.fill" : "checkmark.square",
                    accessibilityLabel: isAllVisibleSelected ? "Clear All Selection" : "Select All Visible"
                ) {
                    if isAllVisibleSelected {
                        clearSelection()
                    } else {
                        selectAll()
                    }
                }
                .accessibilityIdentifier("workspace-selection-toggle-button")

                WorkspaceIconButton(
                    systemName: "trash",
                    accessibilityLabel: "Move Selection to Trash",
                    isDangerous: true
                ) {
                    guard usageState.selectionCount > 0 else {
                        return
                    }
                    deleteConfirmationContext = .selection
                }
                .accessibilityIdentifier("workspace-delete-button")

                WorkspaceIconButton(
                    systemName: "checkmark.circle",
                    accessibilityLabel: "Done Selecting"
                ) {
                    leaveSelectionMode()
                }
                .accessibilityIdentifier("workspace-selection-mode-button")
            } else {
                WorkspaceIconButton(
                    systemName: "checkmark.circle",
                    accessibilityLabel: "Enter Selection Mode"
                ) {
                    enterSelectionMode()
                }
                .accessibilityIdentifier("workspace-selection-mode-button")
            }
        }
    }

    private var settingsButton: some View {
        WorkspaceIconButton(
            systemName: "slider.horizontal.3",
            accessibilityLabel: "Open Settings",
            isProminent: libraryState.selectedFolder == nil
        ) {
            openSettings()
        }
        .accessibilityIdentifier("open-settings-button")
    }

    private var emptyWorkspaceGuide: some View {
        VStack(alignment: .center, spacing: 12) {
            Text("No Selected Folder")
                .font(.system(size: 30, weight: .semibold, design: .serif))
                .foregroundStyle(MuseumPaperTheme.ink)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("workspace-empty-title")

            Text("Open Settings to choose a folder.")
                .font(.system(size: 15))
                .foregroundStyle(MuseumPaperTheme.mutedInk)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("workspace-empty-message")
        }
        .frame(maxWidth: 620)
        .padding(.horizontal, 14)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("workspace-empty-guide")
    }

    private var emptyResultsState: some View {
        let trimmedQuery = usageState.queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let isQueryDrivenEmptyState = trimmedQuery.isEmpty == false || usageState.pastedImageData != nil
        let emptyStateTitle: String
        let emptyStateMessage: String

        if let searchIssue = libraryState.searchIssue, isQueryDrivenEmptyState {
            emptyStateTitle = "Semantic search is unavailable for this folder."
            emptyStateMessage = searchIssue
        } else {
            emptyStateTitle = "Nothing from the current query rose to the surface."
            emptyStateMessage = "Try another phrase, remove the pasted image, or ask for a wider return set."
        }

        return VStack(spacing: 12) {
            Spacer(minLength: 80)

            Text(emptyStateTitle)
                .font(.system(size: 28, weight: .semibold, design: .serif))
                .foregroundStyle(MuseumPaperTheme.ink)

            Text(emptyStateMessage)
                .font(.system(size: 14))
                .foregroundStyle(MuseumPaperTheme.mutedInk)
                .multilineTextAlignment(.center)

            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private func previewOverlay(for item: SearchAsset) -> some View {
        ZStack {
            Color.black.opacity(0.72)
                .ignoresSafeArea()
                .onTapGesture(perform: closePreview)

            WorkspacePreviewKeyboardBridge(
                onEscape: closePreview,
                onMoveLeft: showPreviousPreviewItem,
                onMoveRight: showNextPreviewItem
            )
            .frame(width: 1, height: 1)

            WorkspacePreviewOverlay(
                item: item,
                canShowPrevious: canShowPreviousPreview,
                canShowNext: canShowNextPreview,
                isMetadataVisible: usageState.isPreviewMetadataVisible,
                thumbnailCache: thumbnailCache,
                closePreview: closePreview,
                toggleMetadata: togglePreviewMetadata,
                searchSimilar: searchSimilarToPreview,
                deleteImage: {
                    deleteConfirmationContext = .preview(filename: item.filename)
                },
                showPrevious: showPreviousPreviewItem,
                showNext: showNextPreviewItem
            )

            Color.clear
                .frame(width: 1, height: 1)
                .accessibilityElement()
                .accessibilityLabel("Preview Overlay")
                .accessibilityIdentifier("workspace-preview-overlay")
        }
    }

    private var pastedPreviewImage: NSImage? {
        guard let data = usageState.pastedImageData else {
            return nil
        }
        return NSImage(data: data)
    }

    private var isAllVisibleSelected: Bool {
        usageState.results.isEmpty == false
            && usageState.selectionCount == usageState.results.count
    }

    private var searchAssistText: String? {
        if usageState.isSearching {
            return "Searching locally"
        }

        let hasActiveQuery =
            usageState.queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            || usageState.pastedImageData != nil
        if libraryState.searchIssue != nil, hasActiveQuery {
            return "Semantic search unavailable"
        }

        if usageState.pastedImageData != nil {
            return "Pasted image ready"
        }

        return nil
    }

    private var thumbnailPrefetchIDs: [Int64] {
        Array(usageState.results.prefix(15).map(\.id))
    }

    private var previewIndex: Int? {
        guard let previewResultID = usageState.previewResultID else {
            return nil
        }
        return usageState.results.firstIndex(where: { $0.id == previewResultID })
    }

    private var canShowPreviousPreview: Bool {
        guard let previewIndex else {
            return false
        }
        return previewIndex > 0
    }

    private var canShowNextPreview: Bool {
        guard let previewIndex else {
            return false
        }
        return previewIndex < usageState.results.count - 1
    }

    private var deleteDialogTitle: String {
        switch deleteConfirmationContext {
        case .selection:
            return usageState.selectionCount == 1
                ? "Move Image to Trash?"
                : "Move \(usageState.selectionCount) Images to Trash?"
        case .preview:
            return "Move Image to Trash?"
        case nil:
            return "Move Image to Trash?"
        }
    }

    private var deleteDialogMessage: String {
        switch deleteConfirmationContext {
        case let .preview(filename):
            return "\(filename) will leave this view and move to the Trash."
        case .selection:
            let selectedNames = usageState.results
                .filter { usageState.selectedResultIDs.contains($0.id) }
                .map(\.filename)
                .sorted()

            guard let firstName = selectedNames.first else {
                return "The selected images will leave this view and move to the Trash."
            }

            if selectedNames.count == 1 {
                return "\(firstName) will leave this view and move to the Trash."
            }

            return "\(firstName) and \(selectedNames.count - 1) more images will leave this view and move to the Trash."
        case nil:
            return "The selected images will leave this view and move to the Trash."
        }
    }

    private var deleteDialogBinding: Binding<Bool> {
        Binding(
            get: { deleteConfirmationContext != nil },
            set: { isPresented in
                if isPresented == false {
                    deleteConfirmationContext = nil
                }
            }
        )
    }

    private func performConfirmedDelete() {
        switch deleteConfirmationContext {
        case .selection:
            deleteSelection()
        case .preview:
            deletePreview()
        case nil:
            break
        }
        deleteConfirmationContext = nil
    }

    private var selectionCountBadge: some View {
        Text("\(usageState.selectionCount) selected")
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(MuseumPaperTheme.accentStrong)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                Capsule(style: .continuous)
                    .fill(MuseumPaperTheme.noteFill.opacity(0.92))
                    .overlay(
                        Capsule(style: .continuous)
                            .stroke(MuseumPaperTheme.line, lineWidth: 1)
                    )
            )
    }
}
