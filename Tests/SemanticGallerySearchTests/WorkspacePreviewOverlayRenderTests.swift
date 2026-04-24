import AppKit
import CoreGraphics
import SwiftUI
import Testing
@testable import SemanticGallerySearch

@Test
func workspacePreviewKeepsPortraitMediaVerticallyCenteredInLargeWindows() {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller opens the preview overlay inside a 1920×1080 desktop window.
    ///   2. The selected result is a portrait image with size 640×1600.
    ///   3. Metadata is hidden, so the preview media is the only centered stage inside the overlay.
    /// Postconditions:
    ///   1. The fitted preview media stays within the 1180×760 media bounds.
    ///   2. The fitted media midpoint is equal to the overlay midpoint.

    let frame = WorkspacePreviewLayout.centeredMediaFrame(
        containerSize: CGSize(width: 1920, height: 1080),
        imageSize: CGSize(width: 640, height: 1600)
    )

    #expect(frame.width == 304)
    #expect(frame.height == 760)
    #expect(abs(frame.midY - 540) < 0.5)
}

@Test
func workspacePreviewPinsControlsToTheWindowAndFitsWideImagesInsideTheStage() {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller opens the preview overlay in a 1600×900 window.
    ///   2. The selected result is a wide image with size 4000×1000.
    /// Postconditions:
    ///   1. The media stage remains the fixed caller-visible rectangle inside the preview.
    ///   2. The fitted media frame stays fully inside that stage.
    ///   3. The top-right controls stay anchored to the window corner, not to the media frame.

    let containerSize = CGSize(width: 1600, height: 900)
    let stageFrame = WorkspacePreviewLayout.mediaStageFrame(in: containerSize)
    let mediaFrame = WorkspacePreviewLayout.fittedMediaFrame(
        in: containerSize,
        imageSize: CGSize(width: 4000, height: 1000)
    )
    let controlFrame = WorkspacePreviewLayout.controlBarFrame(
        in: containerSize,
        controlCount: 4
    )

    #expect(stageFrame.width == 1180)
    #expect(stageFrame.height == 760)
    #expect(stageFrame.midX == 800)
    #expect(stageFrame.midY == 450)

    #expect(mediaFrame.minX >= stageFrame.minX)
    #expect(mediaFrame.maxX <= stageFrame.maxX)
    #expect(mediaFrame.minY >= stageFrame.minY)
    #expect(mediaFrame.maxY <= stageFrame.maxY)
    #expect(mediaFrame.width == 1180)
    #expect(mediaFrame.height == 295)

    #expect(controlFrame.maxX == 1574)
    #expect(controlFrame.minY == 26)
}

@MainActor
@Test
func selectableMetadataValueUsesAReadOnlySelectableTextField() throws {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller renders a metadata value inside the preview info card.
    ///   2. The metadata value can span more than one line and may need truncation.
    /// Postconditions:
    ///   1. The rendered AppKit control is selectable so the caller can copy it.
    ///   2. The control remains read-only and borderless.
    ///   3. The line limit matches the caller-visible metadata layout.

    let hostingView = NSHostingView(
        rootView: SelectableMetadataValue(
            text: "/Users/example/Pictures/Folder/File Name.jpg",
            lineLimit: 2,
            truncationMode: .middle,
            accessibilityIdentifier: "workspace-preview-metadata-path-value"
        )
        .frame(width: 320)
    )
    hostingView.frame = NSRect(x: 0, y: 0, width: 320, height: 60)
    hostingView.layoutSubtreeIfNeeded()

    let textView = try #require(findTextView(in: hostingView))
    #expect(textView.isSelectable)
    #expect(textView.isEditable == false)
    #expect(textView.drawsBackground == false)
    #expect(textView.textContainer?.maximumNumberOfLines == 2)
    #expect(textView.identifier?.rawValue == "workspace-preview-metadata-path-value")
}

@MainActor
@Test
func selectableMetadataValueCopiesItsRenderedText() throws {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller renders a metadata value inside the preview info card.
    ///   2. The caller clicks the value and requests Copy.
    /// Postconditions:
    ///   1. The entire metadata value is selected.
    ///   2. Copy writes the exact visible string into the general pasteboard.

    let value = "/Users/example/Pictures/Folder/File Name.jpg"
    let hostingView = NSHostingView(
        rootView: SelectableMetadataValue(
            text: value,
            lineLimit: 2,
            truncationMode: .middle,
            accessibilityIdentifier: "workspace-preview-metadata-path-value"
        )
        .frame(width: 320, height: 60)
    )
    hostingView.frame = NSRect(x: 0, y: 0, width: 320, height: 60)
    hostingView.layoutSubtreeIfNeeded()

    let textView = try #require(findTextView(in: hostingView))
    NSPasteboard.general.clearContents()
    textView.selectAll(nil)
    textView.copy(nil)

    #expect(NSPasteboard.general.string(forType: .string) == value)
}

@MainActor
private func findTextView(in view: NSView) -> NSTextView? {
    if let textView = view as? NSTextView {
        return textView
    }

    for subview in view.subviews {
        if let textView = findTextView(in: subview) {
            return textView
        }
    }

    return nil
}
