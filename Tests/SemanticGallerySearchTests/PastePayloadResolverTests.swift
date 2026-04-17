import AppKit
import Foundation
import Testing
@testable import SemanticGallerySearch

@MainActor
@Test
func pasteAwareSearchTextFieldStartsEditableAndSelectable() {
    let textField = PasteAwareSearchTextField(frame: NSRect(x: 0, y: 0, width: 320, height: 24))

    #expect(textField.isEditable)
    #expect(textField.isSelectable)
}

@MainActor
@Test
func pasteAwareSearchTextFieldRoutesImageFilePasteIntoImageHandler() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let image = try #require(makeImage(color: .systemBlue))
    let imageData = try #require(image.pngData())
    let imageURL = root.appending(path: "sample.png")
    try imageData.write(to: imageURL)

    let pasteboard = NSPasteboard.general
    pasteboard.clearContents()
    #expect(pasteboard.writeObjects([imageURL as NSURL]))

    let textField = PasteAwareSearchTextField(frame: NSRect(x: 0, y: 0, width: 320, height: 24))
    var receivedData: Data?
    textField.onImagePaste = { receivedData = $0 }

    let event = try #require(
        NSEvent.keyEvent(
            with: .keyDown,
            location: .zero,
            modifierFlags: [.command],
            timestamp: 0,
            windowNumber: 0,
            context: nil,
            characters: "v",
            charactersIgnoringModifiers: "v",
            isARepeat: false,
            keyCode: 9
        )
    )

    #expect(textField.performKeyEquivalent(with: event))

    #expect(receivedData == imageData)
    #expect(textField.stringValue.isEmpty)
}

@Test
func pastePayloadResolverLoadsSupportedImageFileData() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let image = try #require(makeImage(color: .systemBlue))
    let imageData = try #require(image.pngData())
    let imageURL = root.appending(path: "sample.png")
    try imageData.write(to: imageURL)

    let resolved = PastePayloadResolver.imageData(from: [imageURL])

    #expect(resolved == imageData)
}

@Test
func pastePayloadResolverRejectsUnsupportedFileURLs() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let textURL = root.appending(path: "notes.txt")
    try Data("hello".utf8).write(to: textURL)

    #expect(PastePayloadResolver.imageData(from: [textURL]) == nil)
}

@Test
func pastePayloadResolverEncodesImagesIntoReadablePNGData() throws {
    let image = try #require(makeImage(color: .systemRed))

    let data = try #require(PastePayloadResolver.imageData(from: [image]))

    #expect(data.isEmpty == false)
    #expect(NSImage(data: data) != nil)
}

private func makeImage(color: NSColor) -> NSImage? {
    let image = NSImage(size: NSSize(width: 8, height: 8))
    image.lockFocus()
    defer { image.unlockFocus() }
    color.drawSwatch(in: NSRect(x: 0, y: 0, width: 8, height: 8))
    return image
}

private extension NSImage {
    func pngData() -> Data? {
        guard
            let tiffRepresentation,
            let bitmap = NSBitmapImageRep(data: tiffRepresentation)
        else {
            return nil
        }
        return bitmap.representation(using: .png, properties: [:])
    }
}
