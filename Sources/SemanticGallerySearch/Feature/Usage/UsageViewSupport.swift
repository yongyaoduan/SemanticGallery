import SwiftUI

enum WorkspaceMetrics {
    static let controlHeight: CGFloat = 44
    static let controlCornerRadius: CGFloat = 16
    static let buttonSize: CGFloat = 44
    static let buttonCornerRadius: CGFloat = 14
}

enum WorkspaceGridLayout {
    static let columnCount = 5
    static let itemSpacing: CGFloat = 6
    static let minimumThumbnailSide: CGFloat = 420
    static let thumbnailOverscan: CGFloat = 1.15

    static func cellSide(forAvailableWidth availableWidth: CGFloat) -> CGFloat {
        let clampedWidth = max(availableWidth, 0)
        let totalSpacing = itemSpacing * CGFloat(columnCount - 1)
        let usableWidth = max(clampedWidth - totalSpacing, 0)
        return floor(usableWidth / CGFloat(columnCount))
    }

    static func thumbnailRequestSize(forAvailableWidth availableWidth: CGFloat) -> CGSize {
        let side = cellSide(forAvailableWidth: availableWidth)
        let requestedSide = ceil(max(minimumThumbnailSide, side * thumbnailOverscan))
        return CGSize(width: requestedSide, height: requestedSide)
    }
}

enum WorkspaceDeleteContext {
    case selection
    case preview(filename: String)
}
