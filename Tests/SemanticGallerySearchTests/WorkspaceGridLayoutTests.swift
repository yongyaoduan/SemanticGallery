import CoreGraphics
import Testing
@testable import SemanticGallerySearch

@Test
func workspaceGridRequestsLargerThumbnailsForWideWindows() {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller renders the fixed 5-column results grid inside a wide window.
    ///   2. The visible grid width is 2400 points.
    /// Postconditions:
    ///   1. The square cell size matches the visible 5-column layout.
    ///   2. The requested thumbnail size grows with the cell size instead of staying at the legacy 420-point cap.

    let cellSide = WorkspaceGridLayout.cellSide(forAvailableWidth: 2400)
    let requestSize = WorkspaceGridLayout.thumbnailRequestSize(forAvailableWidth: 2400)

    #expect(cellSide == 475)
    #expect(requestSize.width == 547)
    #expect(requestSize.height == 547)
}
