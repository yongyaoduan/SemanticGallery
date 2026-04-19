import AppKit
import SwiftUI
import Testing
@testable import SemanticGallerySearch

@MainActor
@Test
func workspacePreviewKeepsPortraitMediaVerticallyCenteredInLargeWindows() throws {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller opens the preview overlay inside a tall desktop window.
    ///   2. The selected result is a portrait image that fits within the preview's maximum media bounds.
    ///   3. Preview metadata is hidden, so the caller expects the media stage to determine the visible center.
    /// Postconditions:
    ///   1. The previewed image remains vertically centered within the overlay.
    ///   2. The caller-visible media midpoint stays close to the window midpoint instead of drifting toward the top edge.

    let fixtureRoot = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: fixtureRoot) }

    let portraitImageURL = fixtureRoot.appending(path: "portrait-preview.png")
    try writeSolidColorImage(
        to: portraitImageURL,
        size: NSSize(width: 640, height: 1600),
        color: .white
    )

    let cacheRoot = fixtureRoot.appending(path: "thumbnail-cache")
    let overlay = WorkspacePreviewOverlay(
        item: SearchAsset(assetID: 7, absolutePath: portraitImageURL.path, thumbnailPath: nil),
        canShowPrevious: true,
        canShowNext: true,
        isMetadataVisible: false,
        thumbnailCache: ThumbnailCache(cacheRoot: cacheRoot),
        closePreview: {},
        toggleMetadata: {},
        searchSimilar: {},
        deleteImage: {},
        showPrevious: {},
        showNext: {}
    )
    .frame(width: 1920, height: 1080)
    .background(Color.black)

    let hostingView = NSHostingView(rootView: overlay)
    hostingView.frame = NSRect(x: 0, y: 0, width: 1920, height: 1080)
    hostingView.layoutSubtreeIfNeeded()
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.35))

    let placement = try previewMediaPlacement(in: hostingView)

    #expect(placement.visiblePixelCount > 50_000)
    #expect(abs(placement.mediaMidY - placement.canvasMidY) < 40)
}

private struct PreviewMediaPlacement {
    let mediaMidY: Double
    let canvasMidY: Double
    let visiblePixelCount: Int
}

@MainActor
private func previewMediaPlacement(in view: NSView) throws -> PreviewMediaPlacement {
    view.layoutSubtreeIfNeeded()
    let bounds = view.bounds.integral
    let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: bounds))
    bitmap.size = bounds.size
    view.cacheDisplay(in: bounds, to: bitmap)

    let width = bitmap.pixelsWide
    let height = bitmap.pixelsHigh
    let bytesPerRow = bitmap.bytesPerRow
    let bytesPerPixel = bitmap.bitsPerPixel / 8
    let data = try #require(bitmap.bitmapData)

    let scanStartX = width / 3
    let scanEndX = (width * 2) / 3
    var minimumY = height
    var maximumY = -1
    var visiblePixelCount = 0

    for y in 0..<height {
        for x in scanStartX..<scanEndX {
            let offset = y * bytesPerRow + x * bytesPerPixel
            let red = Double(data[offset]) / 255.0
            let green = Double(data[offset + 1]) / 255.0
            let blue = Double(data[offset + 2]) / 255.0
            let alpha = Double(data[offset + 3]) / 255.0
            let brightness = max(red, green, blue)

            guard alpha > 0.9, brightness > 0.94 else {
                continue
            }

            visiblePixelCount += 1
            minimumY = min(minimumY, y)
            maximumY = max(maximumY, y)
        }
    }

    #expect(visiblePixelCount > 0)
    #expect(maximumY >= minimumY)

    return PreviewMediaPlacement(
        mediaMidY: Double(minimumY + maximumY) / 2.0,
        canvasMidY: Double(height - 1) / 2.0,
        visiblePixelCount: visiblePixelCount
    )
}

private func makeTemporaryDirectory() throws -> URL {
    let directoryURL = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
    return directoryURL
}

private func writeSolidColorImage(to url: URL, size: NSSize, color: NSColor) throws {
    let bitmap = try #require(
        NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(size.width),
            pixelsHigh: Int(size.height),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        )
    )

    NSGraphicsContext.saveGraphicsState()
    defer { NSGraphicsContext.restoreGraphicsState() }

    let context = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.current = context
    color.setFill()
    NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill()

    let data = try #require(bitmap.representation(using: .png, properties: [:]))
    try data.write(to: url, options: .atomic)
}
