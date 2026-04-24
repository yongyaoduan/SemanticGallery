import AppKit
import SwiftUI
import SemanticGalleryUI

struct WorkspaceIconButton: View {
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

struct WorkspacePreviewOverlay: View {
    let item: SearchAsset
    let canShowPrevious: Bool
    let canShowNext: Bool
    let isMetadataVisible: Bool
    let thumbnailCache: ThumbnailCache
    let closePreview: () -> Void
    let toggleMetadata: () -> Void
    let searchSimilar: () -> Void
    let deleteImage: () -> Void
    let showPrevious: () -> Void
    let showNext: () -> Void

    @State private var image: NSImage?
    @State private var metadata: PreviewMetadata?

    var body: some View {
        GeometryReader { geometry in
            let stageSize = WorkspacePreviewLayout.mediaStageSize(in: geometry.size)

            ZStack {
                HStack {
                    previewArrow(systemName: "chevron.left", enabled: canShowPrevious, action: showPrevious)

                    Spacer(minLength: WorkspacePreviewLayout.arrowSpacing)

                    previewMediaView(stageSize: stageSize)

                    Spacer(minLength: WorkspacePreviewLayout.arrowSpacing)

                    previewArrow(systemName: "chevron.right", enabled: canShowNext, action: showNext)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .padding(.horizontal, WorkspacePreviewLayout.horizontalPadding)

                VStack(alignment: .trailing, spacing: 12) {
                    HStack(spacing: WorkspacePreviewLayout.controlSpacing) {
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
                .padding(WorkspacePreviewLayout.controlPadding)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture {}
        .task(id: item.id) {
            image = nil
            metadata = nil

            if let localImage = NSImage(contentsOf: URL(filePath: item.absolutePath)) {
                image = localImage
            } else {
                image = await thumbnailCache.cachedImage(
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
            metadata = try? PreviewMetadataLoader.load(for: item)
        }
    }

    private func previewMediaView(stageSize: CGSize) -> some View {
        ZStack {
            Color.clear

            if let image {
                let fittedSize = WorkspacePreviewLayout.fittedMediaSize(for: image.size, in: stageSize)
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: fittedSize.width, height: fittedSize.height)
                    .shadow(color: Color.black.opacity(0.28), radius: 26, y: 16)
            } else {
                ProgressView()
                    .controlSize(.large)
                    .tint(.white)
                    .frame(width: 220, height: 220)
            }
        }
        .frame(width: stageSize.width, height: stageSize.height)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Preview Media")
        .accessibilityIdentifier("workspace-preview-media")
    }

    private func previewArrow(systemName: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        WorkspaceIconButton(
            systemName: systemName,
            accessibilityLabel: systemName == "chevron.left" ? "Previous Image" : "Next Image"
        ) {
            action()
        }
        .opacity(enabled ? 1 : 0.32)
        .accessibilityIdentifier(systemName == "chevron.left" ? "workspace-preview-previous-button" : "workspace-preview-next-button")
    }

    private func previewMetadataPanel(_ metadata: PreviewMetadata) -> some View {
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
        let lineLimit = label == "Path" ? 2 : 1

        return VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .bold))
                .tracking(1.1)
                .foregroundStyle(MuseumPaperTheme.mutedInk)

            SelectableMetadataValue(
                text: value,
                lineLimit: lineLimit,
                truncationMode: .middle,
                accessibilityIdentifier: "workspace-preview-metadata-\(label.lowercased())-value"
            )
            .frame(maxWidth: .infinity, minHeight: lineLimit == 1 ? 18 : 36, alignment: .leading)
        }
    }
}

struct SelectableMetadataValue: NSViewRepresentable {
    let text: String
    let lineLimit: Int
    let truncationMode: Text.TruncationMode
    let accessibilityIdentifier: String

    func makeNSView(context: Context) -> SelectableMetadataTextView {
        let textView = SelectableMetadataTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.isRichText = false
        textView.importsGraphics = false
        textView.usesFindBar = false
        textView.allowsUndo = false
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = false
        textView.textContainerInset = .zero
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.font = .systemFont(ofSize: 13, weight: .medium)
        textView.textColor = NSColor(MuseumPaperTheme.ink)
        textView.identifier = NSUserInterfaceItemIdentifier(accessibilityIdentifier)
        updateTextView(textView)
        return textView
    }

    func updateNSView(_ textView: SelectableMetadataTextView, context: Context) {
        updateTextView(textView)
    }

    private func updateTextView(_ textView: SelectableMetadataTextView) {
        textView.string = text
        textView.identifier = NSUserInterfaceItemIdentifier(accessibilityIdentifier)
        textView.textContainer?.maximumNumberOfLines = lineLimit
        textView.textContainer?.lineBreakMode = truncationMode.lineBreakMode
    }
}

final class SelectableMetadataTextView: NSTextView {
    override var acceptsFirstResponder: Bool {
        true
    }

    override func becomeFirstResponder() -> Bool {
        let didBecomeFirstResponder = super.becomeFirstResponder()
        if didBecomeFirstResponder {
            selectAll(nil)
        }
        return didBecomeFirstResponder
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        selectAll(nil)
    }
}

private extension Text.TruncationMode {
    var lineBreakMode: NSLineBreakMode {
        switch self {
        case .head:
            return .byTruncatingHead
        case .middle:
            return .byTruncatingMiddle
        case .tail:
            return .byTruncatingTail
        @unknown default:
            return .byTruncatingTail
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

enum WorkspacePreviewLayout {
    static let maxMediaSize = CGSize(width: 1180, height: 760)
    static let horizontalPadding: CGFloat = 32
    static let controlPadding: CGFloat = 26
    static let controlSpacing: CGFloat = 10
    static let arrowSpacing: CGFloat = 18

    static func mediaStageSize(in containerSize: CGSize) -> CGSize {
        let reservedWidth = horizontalPadding * 2 + (WorkspaceMetrics.buttonSize * 2) + (arrowSpacing * 2)
        let availableWidth = max(containerSize.width - reservedWidth, 220)
        let availableHeight = max(containerSize.height, 220)

        return CGSize(
            width: min(maxMediaSize.width, availableWidth),
            height: min(maxMediaSize.height, availableHeight)
        )
    }

    static func fittedMediaSize(for imageSize: CGSize, in stageSize: CGSize = maxMediaSize) -> CGSize {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return stageSize
        }

        let widthScale = stageSize.width / imageSize.width
        let heightScale = stageSize.height / imageSize.height
        let scale = min(widthScale, heightScale)

        return CGSize(
            width: imageSize.width * scale,
            height: imageSize.height * scale
        )
    }

    static func centeredMediaFrame(containerSize: CGSize, imageSize: CGSize) -> CGRect {
        let fittedSize = fittedMediaSize(for: imageSize, in: mediaStageSize(in: containerSize))
        return CGRect(
            x: (containerSize.width - fittedSize.width) / 2,
            y: (containerSize.height - fittedSize.height) / 2,
            width: fittedSize.width,
            height: fittedSize.height
        )
    }

    static func mediaStageFrame(in containerSize: CGSize) -> CGRect {
        let stageSize = mediaStageSize(in: containerSize)
        return CGRect(
            x: (containerSize.width - stageSize.width) / 2,
            y: (containerSize.height - stageSize.height) / 2,
            width: stageSize.width,
            height: stageSize.height
        )
    }

    static func fittedMediaFrame(in containerSize: CGSize, imageSize: CGSize) -> CGRect {
        let stageFrame = mediaStageFrame(in: containerSize)
        let fittedSize = fittedMediaSize(for: imageSize, in: stageFrame.size)
        return CGRect(
            x: stageFrame.minX + ((stageFrame.width - fittedSize.width) / 2),
            y: stageFrame.minY + ((stageFrame.height - fittedSize.height) / 2),
            width: fittedSize.width,
            height: fittedSize.height
        )
    }

    static func controlBarFrame(in containerSize: CGSize, controlCount: Int) -> CGRect {
        let width = (CGFloat(controlCount) * WorkspaceMetrics.buttonSize) + (CGFloat(max(controlCount - 1, 0)) * controlSpacing)
        return CGRect(
            x: containerSize.width - controlPadding - width,
            y: controlPadding,
            width: width,
            height: WorkspaceMetrics.buttonSize
        )
    }
}

struct WorkspacePreviewKeyboardBridge: NSViewRepresentable {
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

final class WorkspacePreviewKeyView: NSView {
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
