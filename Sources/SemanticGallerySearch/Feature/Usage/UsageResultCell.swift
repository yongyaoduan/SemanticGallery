import AppKit
import SwiftUI
import SemanticGalleryUI

struct UsageResultCell: View {
    let item: SearchAsset
    let isSelected: Bool
    let showsSelection: Bool
    let thumbnailCache: ThumbnailCache
    let thumbnailSize: CGSize
    let action: () -> Void

    @State private var image: NSImage?

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Rectangle()
                    .fill(MuseumPaperTheme.backgroundTop.opacity(0.62))
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        if let image {
                            Image(nsImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        } else {
                            ProgressView()
                                .tint(MuseumPaperTheme.accent)
                        }
                    }
                    .clipped()
                    .overlay {
                        if showsSelection && isSelected {
                            Rectangle()
                                .stroke(MuseumPaperTheme.accentStrong, lineWidth: 3)
                        }
                    }

                if showsSelection && isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundStyle(Color.white, MuseumPaperTheme.accentStrong)
                        .padding(8)
                        .shadow(color: Color.black.opacity(0.18), radius: 10, y: 3)
                }
            }
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("workspace-result-cell")
        .accessibilityLabel(item.relativePath)
        .accessibilityValue(image == nil ? "thumbnail loading" : "thumbnail loaded")
        .task(id: "\(item.id)-\(Int(thumbnailSize.width))x\(Int(thumbnailSize.height))") {
            let loadedImage = await thumbnailCache.cachedImage(
                for: item,
                size: thumbnailSize,
                scale: NSScreen.main?.backingScaleFactor ?? 2
            )
            if image == nil {
                image = loadedImage
            }
        }
    }
}
