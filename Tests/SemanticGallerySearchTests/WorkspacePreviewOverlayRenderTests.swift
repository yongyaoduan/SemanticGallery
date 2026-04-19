import CoreGraphics
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
