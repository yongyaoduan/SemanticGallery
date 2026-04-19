import AppKit
import SwiftUI

struct PasteAwareSearchField: NSViewRepresentable {
    @Binding var text: String
    let placeholder: String
    let focusRequest: Int
    let onSubmit: () -> Void
    let onImagePaste: (Data) -> Void

    func makeCoordinator() -> PasteAwareSearchFieldCoordinator {
        PasteAwareSearchFieldCoordinator(
            text: $text,
            onSubmit: onSubmit,
            onImagePaste: onImagePaste
        )
    }

    func makeNSView(context: Context) -> PasteAwareSearchFieldContainer {
        let container = PasteAwareSearchFieldContainer()
        container.setAccessibilityIdentifier("workspace-search-editor")
        container.focusRequest = focusRequest
        container.textField.stringValue = text
        container.textField.placeholderString = placeholder
        container.textField.setAccessibilityIdentifier("workspace-search-text-field")
        container.textField.setAccessibilityLabel(placeholder)
        container.textField.delegate = context.coordinator
        container.textField.onSubmit = context.coordinator.handleSubmit
        container.textField.onImagePaste = context.coordinator.handleImagePaste
        return container
    }

    func updateNSView(_ nsView: PasteAwareSearchFieldContainer, context: Context) {
        nsView.setAccessibilityIdentifier("workspace-search-editor")
        if nsView.focusRequest != focusRequest {
            nsView.focusRequest = focusRequest
            DispatchQueue.main.async {
                nsView.focusSearchField()
            }
        }
        if nsView.textField.stringValue != text {
            nsView.textField.stringValue = text
        }
        nsView.textField.placeholderString = placeholder
        nsView.textField.delegate = context.coordinator
        nsView.textField.onSubmit = context.coordinator.handleSubmit
        nsView.textField.onImagePaste = context.coordinator.handleImagePaste
        nsView.textField.setAccessibilityLabel(placeholder)
    }
}

final class PasteAwareSearchFieldCoordinator: NSObject, NSTextFieldDelegate {
    @Binding private var text: String
    private let onSubmit: () -> Void
    private let onImagePaste: (Data) -> Void

    init(
        text: Binding<String>,
        onSubmit: @escaping () -> Void,
        onImagePaste: @escaping (Data) -> Void
    ) {
        self._text = text
        self.onSubmit = onSubmit
        self.onImagePaste = onImagePaste
    }

    func controlTextDidChange(_ obj: Notification) {
        guard let field = obj.object as? NSTextField else {
            return
        }
        text = field.stringValue
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            onSubmit()
            return true
        }
        return false
    }

    func handleSubmit() {
        onSubmit()
    }

    func handleImagePaste(_ data: Data) {
        onImagePaste(data)
    }
}

final class PasteAwareSearchFieldContainer: NSView {
    let textField: PasteAwareSearchTextField
    var focusRequest = 0

    override init(frame frameRect: NSRect) {
        let textField = PasteAwareSearchTextField(frame: .zero)
        self.textField = textField
        super.init(frame: frameRect)

        addSubview(textField)
        textField.cell = VerticallyCenteredSearchFieldCell(textCell: "")
        textField.isEditable = true
        textField.isSelectable = true
        textField.isEnabled = true
        textField.isBordered = false
        textField.isBezeled = false
        textField.drawsBackground = false
        textField.focusRingType = .none
        textField.font = .systemFont(ofSize: 16)
        textField.lineBreakMode = .byTruncatingTail
        textField.maximumNumberOfLines = 1
        textField.cell?.usesSingleLineMode = true
        textField.cell?.wraps = false
        textField.cell?.truncatesLastVisibleLine = true
        textField.cell?.isScrollable = true
        textField.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            textField.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 1),
            textField.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -1),
            textField.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func focusSearchField() {
        textField.selectText(nil)
        if let editor = textField.window?.fieldEditor(true, for: textField) as? NSTextView {
            editor.selectedRange = NSRange(location: textField.stringValue.count, length: 0)
        }
    }
}

final class PasteAwareSearchTextField: NSTextField {
    var onSubmit: (() -> Void)?
    var onImagePaste: ((Data) -> Void)?

    override var acceptsFirstResponder: Bool {
        true
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        if event.modifierFlags.contains(.command), event.charactersIgnoringModifiers?.lowercased() == "v" {
            if let imageData = PastePayloadResolver.imageData(from: NSPasteboard.general) {
                onImagePaste?(imageData)
                return true
            }
        }
        return super.performKeyEquivalent(with: event)
    }
}

final class VerticallyCenteredSearchFieldCell: NSTextFieldCell {
    override func drawingRect(forBounds rect: NSRect) -> NSRect {
        let horizontalInset: CGFloat = 2
        let baseRect = rect.insetBy(dx: horizontalInset, dy: 0)
        let cellHeight = (super.cellSize(forBounds: baseRect).height) + 1
        let verticalInset = max(0, floor((baseRect.height - cellHeight) / 2))
        return NSRect(
            x: baseRect.origin.x,
            y: baseRect.origin.y + verticalInset,
            width: baseRect.width,
            height: baseRect.height - (verticalInset * 2)
        )
    }

    override func select(withFrame rect: NSRect, in controlView: NSView, editor textObj: NSText, delegate: Any?, start selStart: Int, length selLength: Int) {
        super.select(withFrame: drawingRect(forBounds: rect), in: controlView, editor: textObj, delegate: delegate, start: selStart, length: selLength)
    }

    override func edit(withFrame rect: NSRect, in controlView: NSView, editor textObj: NSText, delegate: Any?, event: NSEvent?) {
        super.edit(withFrame: drawingRect(forBounds: rect), in: controlView, editor: textObj, delegate: delegate, event: event)
    }
}

enum PastePayloadResolver {
    static func imageData(from pasteboard: NSPasteboard) -> Data? {
        if let fileURLs = pasteboard.readObjects(forClasses: [NSURL.self]) as? [URL],
           let data = imageData(from: fileURLs) {
            return data
        }

        if let images = pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage],
           let data = imageData(from: images) {
            return data
        }

        if let tiffData = pasteboard.data(forType: .tiff),
           let image = NSImage(data: tiffData),
           let encoded = imageData(from: [image]) {
            return encoded
        }

        return nil
    }

    static func imageData(from fileURLs: [URL]) -> Data? {
        fileURLs
            .first(where: isSupportedImageURL(_:))
            .flatMap { try? Data(contentsOf: $0) }
    }

    static func imageData(from images: [NSImage]) -> Data? {
        images.lazy.compactMap(encodedImageData(from:)).first
    }

    private static func encodedImageData(from image: NSImage) -> Data? {
        guard
            let tiffRepresentation = image.tiffRepresentation,
            let bitmap = NSBitmapImageRep(data: tiffRepresentation)
        else {
            return nil
        }

        return bitmap.representation(using: .png, properties: [:]) ?? tiffRepresentation
    }

    private static func isSupportedImageURL(_ url: URL) -> Bool {
        let supportedExtensions = Set(["jpg", "jpeg", "png", "heic", "heif"])
        return supportedExtensions.contains(url.pathExtension.lowercased())
    }
}
