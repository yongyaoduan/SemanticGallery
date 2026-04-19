import AppKit
import CoreGraphics
import Testing
@testable import SemanticGallerySearch

@Test
func thumbnailCachePreservesImageAspectRatioWhenWrappingQuickLookOutput() throws {
    /// Formal specification for callers:
    /// Pre: source image width and height are both positive, and `requestedSize` is square.
    /// Post after `displaySize(for:requestedSize:)`:
    /// the returned rectangle fits inside `requestedSize` and preserves the source aspect ratio.
    let image = try #require(makeImage(width: 240, height: 120))

    let displaySize = ThumbnailCache.displaySize(
        for: image,
        requestedSize: CGSize(width: 420, height: 420)
    )

    #expect(displaySize.width == 420)
    #expect(displaySize.height == 210)
    #expect(displaySize.width / displaySize.height == 2)
}

private func makeImage(width: Int, height: Int) -> CGImage? {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    else {
        return nil
    }
    context.setFillColor(NSColor.systemBlue.cgColor)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
}
