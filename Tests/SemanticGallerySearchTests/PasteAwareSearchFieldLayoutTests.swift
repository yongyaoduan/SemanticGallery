import AppKit
import Testing
@testable import SemanticGallerySearch

@MainActor
@Test
func pasteAwareSearchFieldContainerUsesSingleLineVerticalCentering() {
    let container = PasteAwareSearchFieldContainer(frame: NSRect(x: 0, y: 0, width: 420, height: 44))

    #expect(container.subviews.contains(container.textField))
    #expect(container.textField.isBordered == false)
    #expect(container.textField.cell?.usesSingleLineMode == true)
    #expect(container.textField.cell?.isScrollable == true)
}
