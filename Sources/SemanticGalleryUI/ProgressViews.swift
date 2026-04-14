import SwiftUI

public struct LayeredProgressBar: View {
    public let progress: Double

    public init(progress: Double) {
        self.progress = min(max(progress, 0), 1)
    }

    public var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(MuseumPaperTheme.accent.opacity(0.16))
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [MuseumPaperTheme.accent, MuseumPaperTheme.accent.opacity(0.72)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: proxy.size.width * progress)
            }
        }
        .frame(height: 12)
    }
}
