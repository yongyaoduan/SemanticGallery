import AppKit
import SwiftUI
import SemanticGalleryCore
import SemanticGalleryUI

fileprivate enum WorkspaceMetrics {
    static let controlHeight: CGFloat = 44
    static let controlCornerRadius: CGFloat = 16
    static let buttonSize: CGFloat = 44
    static let buttonCornerRadius: CGFloat = 14
}

fileprivate enum WorkspaceDeleteContext {
    case selection
    case preview(filename: String)
}

public struct UsageView: View {
    @Environment(\.openSettings) private var openSettings
    @Bindable private var libraryStateStore: LibraryStateStore
    @Bindable private var workspaceStateStore: WorkspaceStateStore

    private let thumbnailStore: ThumbnailStore
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

    public init(
        libraryStateStore: LibraryStateStore,
        workspaceStateStore: WorkspaceStateStore,
        thumbnailStore: ThumbnailStore,
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
        self.libraryStateStore = libraryStateStore
        self.workspaceStateStore = workspaceStateStore
        self.thumbnailStore = thumbnailStore
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

            workspaceContent(for: libraryStateStore.selectedFolder)

            if let previewItem = workspaceStateStore.previewItem {
                previewOverlay(for: previewItem)
            }
        }
        .animation(.easeInOut(duration: 0.18), value: workspaceStateStore.isSelectionModeEnabled)
        .animation(.easeInOut(duration: 0.18), value: workspaceStateStore.previewAssetID)
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
            } else if workspaceStateStore.results.isEmpty {
                emptyResultsState
            } else {
                ScrollView {
                    LazyVGrid(
                        columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 5),
                        spacing: 6
                    ) {
                        ForEach(workspaceStateStore.results) { item in
                            UsageResultCell(
                                item: item,
                                isSelected: workspaceStateStore.isSelected(item.id),
                                showsSelection: workspaceStateStore.isSelectionModeEnabled,
                                thumbnailStore: thumbnailStore
                            ) {
                                if workspaceStateStore.isSelectionModeEnabled {
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
                    await thumbnailStore.prefetchImages(
                        for: Array(workspaceStateStore.results.prefix(15)),
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
                .disabled(isWorkspaceReadyForSearch == false)
                .opacity(isWorkspaceReadyForSearch ? 1 : 0.82)

            resultLimitControl

            if isWorkspaceReadyForSearch || workspaceStateStore.isSelectionModeEnabled {
                selectionControls
            }

            settingsButton
        }
        .disabled(isPreparingFolder)
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

    @ViewBuilder
    private var searchEditor: some View {
        if isWorkspaceReadyForSearch {
            activeSearchEditor
        } else {
            inactiveSearchEditor
        }
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
                        workspaceStateStore.pastedImageData = nil
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
                text: $workspaceStateStore.queryText,
                placeholder: "Search with words or paste an image",
                onSubmit: runSearch,
                onImagePaste: { data in
                    workspaceStateStore.queryText = ""
                    workspaceStateStore.pastedImageData = data
                    runSearch()
                }
            )
            .frame(maxWidth: .infinity)

            if workspaceStateStore.isSearching {
                ProgressView()
                    .controlSize(.small)
                    .tint(MuseumPaperTheme.accent)
            }

            if let searchAssistText {
                Text(searchAssistText)
                .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(workspaceStateStore.isSearching ? MuseumPaperTheme.accentStrong : MuseumPaperTheme.mutedInk)
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
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("workspace-search-editor")
    }

    private var inactiveSearchEditor: some View {
        HStack {
            Text("Search with words or paste an image")
                .font(.system(size: 16))
                .foregroundStyle(MuseumPaperTheme.mutedInk.opacity(0.68))
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .frame(height: WorkspaceMetrics.controlHeight, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                .fill(MuseumPaperTheme.panel.opacity(0.78))
        )
        .overlay(
            RoundedRectangle(cornerRadius: WorkspaceMetrics.controlCornerRadius, style: .continuous)
                .stroke(MuseumPaperTheme.line, lineWidth: 1)
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("workspace-search-editor")
    }

    private var resultLimitControl: some View {
        Menu {
            ForEach([25, 50, 100, 250], id: \.self) { value in
                Button {
                    workspaceStateStore.resultLimit = value
                } label: {
                    Text("\(value)")
                }
            }
        } label: {
            HStack(spacing: 7) {
                Text("\(workspaceStateStore.resultLimit)")
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
            if workspaceStateStore.isSelectionModeEnabled {
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
                .disabled(workspaceStateStore.results.isEmpty)
                .accessibilityIdentifier("workspace-selection-toggle-button")

                WorkspaceIconButton(
                    systemName: "trash",
                    accessibilityLabel: "Move Selection to Trash",
                    isDangerous: true
                ) {
                    deleteConfirmationContext = .selection
                }
                .disabled(workspaceStateStore.selectionCount == 0)
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
                .disabled(workspaceStateStore.results.isEmpty || isWorkspaceReadyForSearch == false)
                .accessibilityIdentifier("workspace-selection-mode-button")
            }
        }
    }

    private var settingsButton: some View {
        WorkspaceIconButton(
            systemName: "slider.horizontal.3",
            accessibilityLabel: "Open Settings",
            isProminent: libraryStateStore.selectedFolder == nil
        ) {
            openSettings()
        }
        .accessibilityIdentifier("open-settings-button")
    }

    private var emptyWorkspaceGuide: some View {
        VStack(alignment: .center, spacing: 12) {
            Text("Choose a folder to begin")
                .font(.system(size: 30, weight: .semibold, design: .serif))
                .foregroundStyle(MuseumPaperTheme.ink)
                .fixedSize(horizontal: false, vertical: true)

            Text("Open Settings at the far right, then choose your folder")
                .font(.system(size: 15))
                .foregroundStyle(MuseumPaperTheme.mutedInk)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 620)
        .padding(.horizontal, 14)
    }

    private var emptyResultsState: some View {
        let trimmedQuery = workspaceStateStore.queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let isQueryDrivenEmptyState = trimmedQuery.isEmpty == false || workspaceStateStore.pastedImageData != nil
        let emptyStateTitle: String
        let emptyStateMessage: String

        if let searchIssue = libraryStateStore.searchIssue, isQueryDrivenEmptyState {
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

    private func previewOverlay(for item: SearchAssetRecord) -> some View {
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
                isMetadataVisible: workspaceStateStore.isPreviewMetadataVisible,
                thumbnailStore: thumbnailStore,
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
        guard let data = workspaceStateStore.pastedImageData else {
            return nil
        }
        return NSImage(data: data)
    }

    private var isPreparingFolder: Bool {
        libraryStateStore.folderPreparationProgress.isEmpty == false
    }

    private var isWorkspaceReadyForSearch: Bool {
        libraryStateStore.selectedFolder != nil && isPreparingFolder == false
    }

    private var isAllVisibleSelected: Bool {
        workspaceStateStore.results.isEmpty == false
            && workspaceStateStore.selectionCount == workspaceStateStore.results.count
    }

    private var searchAssistText: String? {
        if workspaceStateStore.isSearching {
            return "Searching locally"
        }

        let hasActiveQuery =
            workspaceStateStore.queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            || workspaceStateStore.pastedImageData != nil
        if libraryStateStore.searchIssue != nil, hasActiveQuery {
            return "Semantic search unavailable"
        }

        if workspaceStateStore.pastedImageData != nil {
            return "Pasted image ready"
        }

        return nil
    }

    private var thumbnailPrefetchIDs: [Int64] {
        Array(workspaceStateStore.results.prefix(15).map(\.id))
    }

    private var previewIndex: Int? {
        guard let previewAssetID = workspaceStateStore.previewAssetID else {
            return nil
        }
        return workspaceStateStore.results.firstIndex(where: { $0.id == previewAssetID })
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
        return previewIndex < workspaceStateStore.results.count - 1
    }

    private var deleteDialogTitle: String {
        switch deleteConfirmationContext {
        case .selection:
            return workspaceStateStore.selectionCount == 1
                ? "Move Image to Trash?"
                : "Move \(workspaceStateStore.selectionCount) Images to Trash?"
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
            let selectedNames = workspaceStateStore.results
                .filter { workspaceStateStore.selectedAssetIDs.contains($0.id) }
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
        Text("\(workspaceStateStore.selectionCount) selected")
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

private struct UsageResultCell: View {
    let item: SearchAssetRecord
    let isSelected: Bool
    let showsSelection: Bool
    let thumbnailStore: ThumbnailStore
    let action: () -> Void

    @State private var image: NSImage?

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Rectangle()
                    .fill(MuseumPaperTheme.backgroundTop.opacity(0.62))
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        if let image {
                            Image(nsImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } else {
                            ProgressView()
                                .tint(MuseumPaperTheme.accent)
                        }
                    }
                    .clipped()
                    .overlay {
                        if showsSelection && isSelected {
                            Rectangle()
                                .stroke(MuseumPaperTheme.accentStrong, lineWidth: 3)
                        }
                    }

                if showsSelection && isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundStyle(Color.white, MuseumPaperTheme.accentStrong)
                        .padding(8)
                        .shadow(color: Color.black.opacity(0.18), radius: 10, y: 3)
                }
            }
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("workspace-result-cell")
        .accessibilityLabel(item.relativePath)
        .accessibilityValue(image == nil ? "thumbnail loading" : "thumbnail loaded")
        .task(id: item.id) {
            let loadedImage = await thumbnailStore.cachedImage(
                for: item,
                size: CGSize(width: 420, height: 420),
                scale: NSScreen.main?.backingScaleFactor ?? 2
            )
            if image == nil {
                image = loadedImage
            }
        }
    }
}

private struct WorkspaceIconButton: View {
    let systemName: String
    let accessibilityLabel: String
    var isProminent: Bool = false
    var isDangerous: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(foregroundColor)
                .frame(width: WorkspaceMetrics.buttonSize, height: WorkspaceMetrics.buttonSize)
                .background(
                    RoundedRectangle(cornerRadius: WorkspaceMetrics.buttonCornerRadius, style: .continuous)
                        .fill(backgroundColor)
                        .overlay(
                            RoundedRectangle(cornerRadius: WorkspaceMetrics.buttonCornerRadius, style: .continuous)
                                .stroke(borderColor, lineWidth: 1)
                        )
                )
                .shadow(color: shadowColor, radius: isProminent ? 14 : 6, y: isProminent ? 8 : 3)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }

    private var backgroundColor: Color {
        if isDangerous {
            return MuseumPaperTheme.noteInk
        }
        if isProminent {
            return MuseumPaperTheme.accent
        }
        return MuseumPaperTheme.panel.opacity(0.78)
    }

    private var foregroundColor: Color {
        (isProminent || isDangerous) ? .white : MuseumPaperTheme.accentStrong
    }

    private var borderColor: Color {
        (isProminent || isDangerous) ? Color.white.opacity(0.18) : MuseumPaperTheme.line.opacity(0.92)
    }

    private var shadowColor: Color {
        isProminent ? MuseumPaperTheme.accentStrong.opacity(0.18) : Color.black.opacity(0.08)
    }
}

private struct WorkspacePreviewOverlay: View {
    let item: SearchAssetRecord
    let canShowPrevious: Bool
    let canShowNext: Bool
    let isMetadataVisible: Bool
    let thumbnailStore: ThumbnailStore
    let closePreview: () -> Void
    let toggleMetadata: () -> Void
    let searchSimilar: () -> Void
    let deleteImage: () -> Void
    let showPrevious: () -> Void
    let showNext: () -> Void

    @State private var image: NSImage?
    @State private var metadata: WorkspacePreviewMetadata?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            HStack {
                previewArrow(systemName: "chevron.left", enabled: canShowPrevious, action: showPrevious)

                Spacer(minLength: 18)

                ZStack {
                    if let image {
                        Image(nsImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 1180, maxHeight: 760)
                            .shadow(color: Color.black.opacity(0.28), radius: 26, y: 16)
                    } else {
                        ProgressView()
                            .controlSize(.large)
                            .tint(.white)
                            .frame(width: 220, height: 220)
                    }
                }

                Spacer(minLength: 18)

                previewArrow(systemName: "chevron.right", enabled: canShowNext, action: showNext)
            }
            .padding(.horizontal, 32)

            VStack(alignment: .trailing, spacing: 12) {
                HStack(spacing: 10) {
                    WorkspaceIconButton(
                        systemName: "sparkles",
                        accessibilityLabel: "Search Similar Images"
                    ) {
                        searchSimilar()
                    }
                    .accessibilityIdentifier("workspace-preview-similar-button")

                    WorkspaceIconButton(
                        systemName: "info.circle",
                        accessibilityLabel: "Image Information"
                    ) {
                        toggleMetadata()
                    }
                    .accessibilityIdentifier("workspace-preview-info-button")

                    WorkspaceIconButton(
                        systemName: "trash",
                        accessibilityLabel: "Move Image to Trash",
                        isDangerous: true
                    ) {
                        deleteImage()
                    }
                    .accessibilityIdentifier("workspace-preview-delete-button")

                    WorkspaceIconButton(
                        systemName: "xmark",
                        accessibilityLabel: "Close Preview"
                    ) {
                        closePreview()
                    }
                    .accessibilityIdentifier("workspace-preview-close-button")
                }

                if isMetadataVisible, let metadata {
                    previewMetadataPanel(metadata)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            .padding(26)
        }
        .contentShape(Rectangle())
        .onTapGesture {}
        .task(id: item.id) {
            image = nil
            metadata = nil

            if let localImage = NSImage(contentsOf: URL(filePath: item.absolutePath)) {
                image = localImage
            } else {
                image = await thumbnailStore.cachedImage(
                    for: item,
                    size: CGSize(width: 1280, height: 1280),
                    scale: NSScreen.main?.backingScaleFactor ?? 2
                )
            }
        }
        .task(id: "\(item.id)-\(isMetadataVisible)") {
            guard isMetadataVisible else {
                metadata = nil
                return
            }
            metadata = try? WorkspacePreviewMetadataLoader.load(for: item)
        }
    }

    private func previewArrow(systemName: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        WorkspaceIconButton(
            systemName: systemName,
            accessibilityLabel: systemName == "chevron.left" ? "Previous Image" : "Next Image"
        ) {
            action()
        }
        .disabled(enabled == false)
        .opacity(enabled ? 1 : 0.32)
        .accessibilityIdentifier(systemName == "chevron.left" ? "workspace-preview-previous-button" : "workspace-preview-next-button")
    }

    private func previewMetadataPanel(_ metadata: WorkspacePreviewMetadata) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(metadata.filename)
                .font(.system(size: 18, weight: .semibold, design: .serif))
                .foregroundStyle(MuseumPaperTheme.ink)

            metadataRow(label: "Path", value: metadata.fullPath)
            metadataRow(label: "Captured", value: metadata.captureDate.map(WorkspacePreviewFormatting.captureDate) ?? "Unavailable")
            metadataRow(label: "Size", value: WorkspacePreviewFormatting.fileSize(metadata.fileSize))
            metadataRow(label: "Dimensions", value: WorkspacePreviewFormatting.dimensions(width: metadata.pixelWidth, height: metadata.pixelHeight))
        }
        .padding(18)
        .frame(width: 360, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(MuseumPaperTheme.panel.opacity(0.96))
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(MuseumPaperTheme.line, lineWidth: 1)
                )
        )
        .shadow(color: Color.black.opacity(0.18), radius: 22, y: 10)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("workspace-preview-metadata")
    }

    private func metadataRow(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .bold))
                .tracking(1.1)
                .foregroundStyle(MuseumPaperTheme.mutedInk)

            Text(value)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(MuseumPaperTheme.ink)
                .lineLimit(label == "Path" ? 2 : 1)
                .truncationMode(.middle)
        }
    }
}

private enum WorkspacePreviewFormatting {
    private static let captureFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    static func captureDate(_ date: Date) -> String {
        captureFormatter.string(from: date)
    }

    static func fileSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    static func dimensions(width: Int, height: Int) -> String {
        "\(width) × \(height)"
    }
}

private struct WorkspacePreviewKeyboardBridge: NSViewRepresentable {
    let onEscape: () -> Void
    let onMoveLeft: () -> Void
    let onMoveRight: () -> Void

    func makeNSView(context: Context) -> WorkspacePreviewKeyView {
        let view = WorkspacePreviewKeyView()
        view.onEscape = onEscape
        view.onMoveLeft = onMoveLeft
        view.onMoveRight = onMoveRight
        DispatchQueue.main.async {
            view.window?.makeFirstResponder(view)
        }
        return view
    }

    func updateNSView(_ nsView: WorkspacePreviewKeyView, context: Context) {
        nsView.onEscape = onEscape
        nsView.onMoveLeft = onMoveLeft
        nsView.onMoveRight = onMoveRight
        DispatchQueue.main.async {
            nsView.window?.makeFirstResponder(nsView)
        }
    }
}

private final class WorkspacePreviewKeyView: NSView {
    var onEscape: (() -> Void)?
    var onMoveLeft: (() -> Void)?
    var onMoveRight: (() -> Void)?

    override var acceptsFirstResponder: Bool {
        true
    }

    override func keyDown(with event: NSEvent) {
        switch event.keyCode {
        case 53:
            onEscape?()
        case 123:
            onMoveLeft?()
        case 124:
            onMoveRight?()
        default:
            super.keyDown(with: event)
        }
    }
}
