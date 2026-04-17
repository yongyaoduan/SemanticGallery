import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct WorkspacePreviewMetadata: Equatable, Sendable {
    public let filename: String
    public let fullPath: String
    public let captureDate: Date?
    public let fileSize: Int64
    public let pixelWidth: Int
    public let pixelHeight: Int

    public init(
        filename: String,
        fullPath: String,
        captureDate: Date?,
        fileSize: Int64,
        pixelWidth: Int,
        pixelHeight: Int
    ) {
        self.filename = filename
        self.fullPath = fullPath
        self.captureDate = captureDate
        self.fileSize = fileSize
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
    }
}

public enum WorkspacePreviewMetadataLoader {
    public static func load(for item: SearchAssetRecord) throws -> WorkspacePreviewMetadata {
        try load(fileURL: URL(filePath: item.absolutePath))
    }

    public static func load(fileURL: URL) throws -> WorkspacePreviewMetadata {
        let resourceValues = try fileURL.resourceValues(forKeys: [
            .fileSizeKey,
            .creationDateKey,
            .contentModificationDateKey,
        ])
        let fileAttributes = try FileManager.default.attributesOfItem(atPath: fileURL.path(percentEncoded: false))
        let captureDate = imageCaptureDate(fileURL: fileURL)
            ?? resourceValues.creationDate
            ?? resourceValues.contentModificationDate
            ?? fileAttributes[.creationDate] as? Date
            ?? fileAttributes[.modificationDate] as? Date

        let dimensions = imageDimensions(fileURL: fileURL)

        return WorkspacePreviewMetadata(
            filename: fileURL.lastPathComponent,
            fullPath: fileURL.path(percentEncoded: false),
            captureDate: captureDate,
            fileSize: Int64(resourceValues.fileSize ?? 0),
            pixelWidth: dimensions.width,
            pixelHeight: dimensions.height
        )
    }

    private static func imageCaptureDate(fileURL: URL) -> Date? {
        guard let properties = imageProperties(fileURL: fileURL) else {
            return nil
        }

        if
            let exif = properties[kCGImagePropertyExifDictionary as String] as? [String: Any],
            let dateString = exif[kCGImagePropertyExifDateTimeOriginal as String] as? String,
            let date = parseExifDate(dateString)
        {
            return date
        }

        if
            let tiff = properties[kCGImagePropertyTIFFDictionary as String] as? [String: Any],
            let dateString = tiff[kCGImagePropertyTIFFDateTime as String] as? String,
            let date = parseExifDate(dateString)
        {
            return date
        }

        return nil
    }

    private static func imageDimensions(fileURL: URL) -> (width: Int, height: Int) {
        guard let properties = imageProperties(fileURL: fileURL) else {
            return (0, 0)
        }

        let width = properties[kCGImagePropertyPixelWidth as String] as? Int ?? 0
        let height = properties[kCGImagePropertyPixelHeight as String] as? Int ?? 0
        return (width, height)
    }

    private static func imageProperties(fileURL: URL) -> [String: Any]? {
        guard
            let imageSource = CGImageSourceCreateWithURL(fileURL as CFURL, [
                kCGImageSourceShouldCache: false,
                kCGImageSourceTypeIdentifierHint: UTType.image.identifier as CFString,
            ] as CFDictionary),
            let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [String: Any]
        else {
            return nil
        }

        return properties
    }

    private static func parseExifDate(_ rawValue: String) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy:MM:dd HH:mm:ss"
        return formatter.date(from: rawValue)
    }
}
