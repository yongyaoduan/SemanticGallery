import SwiftUI

public struct MuseumActionButton: View {
    private let title: String
    private let accessibilityIdentifier: String?
    private let action: () -> Void

    @Environment(\.controlSize) private var controlSize
    @Environment(\.isEnabled) private var isEnabled

    public init(
        _ title: String,
        accessibilityIdentifier: String? = nil,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.accessibilityIdentifier = accessibilityIdentifier
        self.action = action
    }

    public var body: some View {
        Group {
            if isEnabled {
                Button(action: action) {
                    buttonLabel
                }
                .buttonStyle(.plain)
            } else {
                buttonLabel
                    .allowsHitTesting(false)
                    .accessibilityElement()
                    .accessibilityAddTraits(.isButton)
            }
        }
        .accessibilityIdentifier(accessibilityIdentifier ?? "")
    }

    private var buttonLabel: some View {
        Text(title)
            .font(.system(size: metrics.fontSize, weight: .semibold))
            .foregroundStyle(isEnabled ? Color.white.opacity(0.98) : Color.white.opacity(0.86))
            .padding(.horizontal, metrics.horizontalPadding)
            .frame(minHeight: metrics.minHeight)
            .background(background)
            .clipShape(shape)
            .shadow(
                color: MuseumPaperTheme.accentStrong.opacity(isEnabled ? 0.12 : 0.04),
                radius: isEnabled ? 2 : 1,
                y: 1
            )
    }

    private var background: some View {
        shape
            .fill(
                LinearGradient(
                    colors: backgroundColors,
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .overlay(alignment: .top) {
                shape
                    .fill(Color.white.opacity(isEnabled ? 0.10 : 0.05))
                    .padding(1)
                    .mask(
                        LinearGradient(
                            colors: [Color.white, Color.white.opacity(0)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
            }
    }

    private var backgroundColors: [Color] {
        if isEnabled == false {
            return [MuseumPaperTheme.actionGoldDisabled, MuseumPaperTheme.actionGoldDisabledEdge]
        }
        return [MuseumPaperTheme.actionGold, MuseumPaperTheme.actionGold.opacity(0.98)]
    }

    private var shape: RoundedRectangle {
        RoundedRectangle(cornerRadius: metrics.cornerRadius, style: .continuous)
    }

    private var metrics: MuseumActionButtonMetrics {
        switch controlSize {
        case .mini:
            return MuseumActionButtonMetrics(minHeight: 24, horizontalPadding: 11, cornerRadius: 8, fontSize: 12)
        case .small:
            return MuseumActionButtonMetrics(minHeight: 28, horizontalPadding: 13, cornerRadius: 9, fontSize: 13)
        case .large:
            return MuseumActionButtonMetrics(minHeight: 36, horizontalPadding: 20, cornerRadius: 11, fontSize: 14)
        case .regular, .extraLarge:
            return MuseumActionButtonMetrics(minHeight: 30, horizontalPadding: 16, cornerRadius: 10, fontSize: 13)
        @unknown default:
            return MuseumActionButtonMetrics(minHeight: 30, horizontalPadding: 16, cornerRadius: 10, fontSize: 13)
        }
    }
}

private struct MuseumActionButtonMetrics {
    let minHeight: CGFloat
    let horizontalPadding: CGFloat
    let cornerRadius: CGFloat
    let fontSize: CGFloat
}
