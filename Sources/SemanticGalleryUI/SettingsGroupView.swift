import SwiftUI

public struct SettingsGroupView<Content: View>: View {
    private let title: String
    private let content: Content

    public init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title.uppercased())
                .font(.system(size: 12, weight: .semibold, design: .default))
                .tracking(1.1)
                .foregroundStyle(MuseumPaperTheme.mutedInk)

            VStack(alignment: .leading, spacing: 12) {
                content
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(MuseumPaperTheme.panel)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .stroke(MuseumPaperTheme.line, lineWidth: 1)
                    )
                    .shadow(color: MuseumPaperTheme.ink.opacity(0.06), radius: 18, y: 10)
            )
        }
    }
}
