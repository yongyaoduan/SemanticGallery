import SwiftUI

public struct LayeredProgressBar: View {
    public let progress: Double
    public let isActive: Bool

    public init(progress: Double, isActive: Bool = false) {
        self.progress = min(max(progress, 0), 1)
        self.isActive = isActive
    }

    public var body: some View {
        TimelineView(.animation(minimumInterval: 0.25)) { timeline in
            GeometryReader { proxy in
                let baseWidth = proxy.size.width * progress
                let nowValue = timeline.date.timeIntervalSinceReferenceDate
                let shimmerOffset = isActive ? CGFloat((nowValue.truncatingRemainder(dividingBy: 1.8)) / 1.8) : 0

                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(MuseumPaperTheme.accent.opacity(0.16))

                    if baseWidth > 0 {
                        Capsule()
                            .fill(
                                LinearGradient(
                                    colors: [MuseumPaperTheme.accent, MuseumPaperTheme.accent.opacity(0.72)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: baseWidth)
                            .overlay(alignment: .leading) {
                                if isActive {
                                    Capsule()
                                        .fill(
                                            LinearGradient(
                                                colors: [
                                                    Color.white.opacity(0.02),
                                                    Color.white.opacity(0.48),
                                                    Color.white.opacity(0.02),
                                                ],
                                                startPoint: .leading,
                                                endPoint: .trailing
                                            )
                                        )
                                        .frame(width: min(max(baseWidth * 0.28, 26), 72))
                                        .offset(x: max(0, baseWidth - min(max(baseWidth * 0.28, 26), 72)) * shimmerOffset)
                                }
                            }
                    } else if isActive {
                        Capsule()
                            .fill(MuseumPaperTheme.accent.opacity(0.22))
                            .frame(width: min(proxy.size.width * 0.12, 44))
                            .offset(x: max(0, proxy.size.width - min(proxy.size.width * 0.12, 44)) * shimmerOffset)
                    }
                }
            }
        }
        .frame(height: 12)
    }
}
