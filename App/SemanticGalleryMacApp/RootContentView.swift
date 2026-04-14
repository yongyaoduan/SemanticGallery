import SwiftUI
import SemanticGalleryCore
import SemanticGalleryInstall
import SemanticGalleryUI

struct RootContentView: View {
    @Bindable var statusStore: AppStatusStore

    var body: some View {
        Group {
            switch statusStore.status {
            case .installRequired, .installing, .installFailed, .installComplete, .uninstalling:
                LaunchFlowView(statusStore: statusStore)
            case .readyWithoutFolder, .preparingFolder, .ready:
                workspacePlaceholder
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var workspacePlaceholder: some View {
        ZStack {
            LinearGradient(
                colors: [MuseumPaperTheme.backgroundTop, MuseumPaperTheme.backgroundBottom],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 16) {
                Text("Choose a folder to continue.")
                    .font(.system(size: 30, weight: .semibold, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.ink)
                Text("The search workspace appears after folder selection. Settings also opens the folder picker.")
                    .foregroundStyle(MuseumPaperTheme.mutedInk)
            }
            .padding(32)
            .background(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(MuseumPaperTheme.panel)
            )
        }
    }
}
