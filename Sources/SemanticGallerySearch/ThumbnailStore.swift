import AppKit
import Foundation
import QuickLookThumbnailing

public actor ThumbnailStore {
    private let cacheRoot: URL
    private let imageCache = NSCache<NSString, NSImage>()
    private var inFlightLoads: [String: Task<NSImage?, Never>] = [:]

    public init(cacheRoot: URL) {
        self.cacheRoot = cacheRoot
        imageCache.countLimit = 240
    }

    public func cachedImage(
        for item: SearchAssetRecord,
        size: CGSize,
        scale: CGFloat
    ) async -> NSImage? {
        let cacheKey = cacheKey(for: item, size: size, scale: scale)
        if let image = imageCache.object(forKey: cacheKey as NSString) {
            return image
        }

        if let existingTask = inFlightLoads[cacheKey] {
            return await existingTask.value
        }

        let task = Task<NSImage?, Never> { [cacheRoot] in
            await Self.loadImage(
                for: item,
                cacheRoot: cacheRoot,
                cacheKey: cacheKey,
                size: size,
                scale: scale
            )
        }
        inFlightLoads[cacheKey] = task

        let image = await task.value
        inFlightLoads.removeValue(forKey: cacheKey)

        if let image {
            imageCache.setObject(image, forKey: cacheKey as NSString)
        }

        return image
    }

    public func prefetchImages(
        for items: [SearchAssetRecord],
        size: CGSize,
        scale: CGFloat,
        limit: Int = 15
    ) async {
        for item in items.prefix(limit) {
            _ = await cachedImage(for: item, size: size, scale: scale)
        }
    }

    private static func loadImage(
        for item: SearchAssetRecord,
        cacheRoot: URL,
        cacheKey: String,
        size: CGSize,
        scale: CGFloat
    ) async -> NSImage? {
        let cacheURL = cacheRoot
            .appending(path: cacheKey)
            .appendingPathExtension("jpg")

        if let image = NSImage(contentsOf: cacheURL) {
            return image
        }

        do {
            try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
            let request = QLThumbnailGenerator.Request(
                fileAt: URL(filePath: item.absolutePath),
                size: size,
                scale: scale,
                representationTypes: .thumbnail
            )
            let representation = try await QLThumbnailGenerator.shared.generateBestRepresentation(for: request)
            let image = NSImage(cgImage: representation.cgImage, size: NSSize(width: size.width, height: size.height))
            if let data = jpegData(from: representation.cgImage) {
                try? data.write(to: cacheURL, options: .atomic)
            }
            return image
        } catch {
            return NSImage(contentsOf: URL(filePath: item.absolutePath))
        }
    }

    private func cacheKey(for item: SearchAssetRecord, size: CGSize, scale: CGFloat) -> String {
        "\(item.assetID)-\(Int(size.width))x\(Int(size.height))@\(Int(scale * 100))"
    }

    private static func jpegData(from image: CGImage) -> Data? {
        let representation = NSBitmapImageRep(cgImage: image)
        return representation.representation(using: .jpeg, properties: [.compressionFactor: 0.88])
    }
}
