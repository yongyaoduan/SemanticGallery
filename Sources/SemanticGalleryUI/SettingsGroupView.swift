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
                .foregroundStyle(MuseumPaperTheme.mutedInk)

            VStack(alignment: .leading, spacing: 12) {
                content
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(MuseumPaperTheme.panel)
            )
        }
    }
}
