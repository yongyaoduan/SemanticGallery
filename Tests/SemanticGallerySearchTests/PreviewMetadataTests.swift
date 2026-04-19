import AppKit
import Foundation
import Testing
@testable import SemanticGallerySearch

@Test
func previewMetadataLoadsPathDimensionsSizeAndFallbackDate() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let fileURL = root.appending(path: "preview-sample.png")
    let imageData = try #require(makePNGData(width: 4, height: 3))
    try imageData.write(to: fileURL)

    let fallbackDate = Date(timeIntervalSince1970: 1_715_000_000)
    try FileManager.default.setAttributes(
        [.modificationDate: fallbackDate],
        ofItemAtPath: fileURL.path(percentEncoded: false)
    )

    let metadata = try PreviewMetadataLoader.load(fileURL: fileURL)

    #expect(metadata.filename == "preview-sample.png")
    #expect(metadata.fullPath == fileURL.path(percentEncoded: false))
    #expect(metadata.pixelWidth == 4)
    #expect(metadata.pixelHeight == 3)
    #expect(metadata.fileSize == Int64(imageData.count))
    #expect(metadata.captureDate?.timeIntervalSince1970 == fallbackDate.timeIntervalSince1970)
}

private func makePNGData(width: Int, height: Int) -> Data? {
    guard
        let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        )
    else {
        return nil
    }

    for x in 0 ..< width {
        for y in 0 ..< height {
            bitmap.setColor(NSColor(calibratedRed: CGFloat(x + 1) / CGFloat(width + 1), green: CGFloat(y + 1) / CGFloat(height + 1), blue: 0.35, alpha: 1), atX: x, y: y)
        }
    }

    return bitmap.representation(using: .png, properties: [:])
}
