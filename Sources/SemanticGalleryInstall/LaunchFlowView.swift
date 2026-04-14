import SwiftUI
import SemanticGalleryCore
import SemanticGalleryUI

public enum LaunchFlowScreen: Equatable {
    case intro
    case progress
    case completion

    public init(status: AppStatus) {
        switch status {
        case .installRequired, .installFailed:
            self = .intro
        case .installing, .uninstalling:
            self = .progress
        case .installComplete:
            self = .completion
        case .readyWithoutFolder, .preparingFolder, .ready:
            self = .intro
        }
    }
}

public struct LaunchFlowView: View {
    @Bindable private var statusStore: AppStatusStore

    public init(statusStore: AppStatusStore) {
        self.statusStore = statusStore
    }

    public var body: some View {
        ZStack {
            LinearGradient(
                colors: [MuseumPaperTheme.backgroundTop, MuseumPaperTheme.backgroundBottom],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            switch LaunchFlowScreen(status: statusStore.status) {
            case .intro:
                LaunchIntroView {
                    statusStore.status = .installing
                }
            case .progress:
                LaunchProgressView {
                    statusStore.status = .installComplete
                }
            case .completion:
                LaunchCompletionView {
                    statusStore.status = .readyWithoutFolder
                }
            }
        }
    }
}

private struct LaunchShell<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 28) {
            content
        }
        .padding(40)
        .frame(maxWidth: 860)
        .background(
            RoundedRectangle(cornerRadius: 32, style: .continuous)
                .fill(MuseumPaperTheme.panel)
                .shadow(color: .black.opacity(0.08), radius: 28, y: 18)
        )
        .padding(48)
    }
}

private struct LaunchIntroView: View {
    let startInstallation: () -> Void

    var body: some View {
        LaunchShell {
            VStack(alignment: .leading, spacing: 18) {
                Text("SemanticGallery")
                    .font(.system(size: 18, weight: .medium, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.mutedInk)

                Text("Install a local semantic photo library.")
                    .font(.system(size: 42, weight: .semibold, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.ink)

                Text("SemanticGallery prepares a local workspace, the SigLIP2 model, the Stage 1 retrieval weights, and the public anchor dataset before you start searching.")
                    .font(.system(size: 18))
                    .foregroundStyle(MuseumPaperTheme.mutedInk)
                    .frame(maxWidth: 620, alignment: .leading)

                Button("Start Installation", action: startInstallation)
                    .buttonStyle(.borderedProminent)
                    .tint(MuseumPaperTheme.accent)
                    .controlSize(.large)
                    .accessibilityIdentifier("start-installation-button")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct LaunchProgressView: View {
    let finishInstallation: () -> Void

    var body: some View {
        LaunchShell {
            VStack(alignment: .leading, spacing: 22) {
                Text("Installing SemanticGallery")
                    .font(.system(size: 38, weight: .semibold, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.ink)

                Text("The install screen and the settings progress views share the same visual language.")
                    .foregroundStyle(MuseumPaperTheme.mutedInk)

                LayeredProgressBar(progress: 0.66)
                    .frame(height: 12)

                VStack(spacing: 14) {
                    progressRow(title: "Prepare directories", progress: 1.0, caption: "Done")
                    progressRow(title: "Download base model", progress: 0.82, caption: "2m left")
                    progressRow(title: "Download public anchor", progress: 0.24, caption: "Queued")
                }

                Button("Mark Installation Complete", action: finishInstallation)
                    .buttonStyle(.borderedProminent)
                    .tint(MuseumPaperTheme.accent)
                    .controlSize(.large)
            }
            .accessibilityIdentifier("launch-progress-screen")
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func progressRow(title: String, progress: Double, caption: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .foregroundStyle(MuseumPaperTheme.ink)
                Spacer()
                Text(caption)
                    .foregroundStyle(MuseumPaperTheme.mutedInk)
            }
            LayeredProgressBar(progress: progress)
                .frame(height: 10)
        }
    }
}

private struct LaunchCompletionView: View {
    let enterApp: () -> Void

    var body: some View {
        LaunchShell {
            VStack(alignment: .leading, spacing: 18) {
                Text("Installation Complete")
                    .font(.system(size: 40, weight: .semibold, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.ink)

                Text("SemanticGallery is ready. Continue into the workspace and choose the folder you want to index.")
                    .foregroundStyle(MuseumPaperTheme.mutedInk)

                Button("Start Using SemanticGallery", action: enterApp)
                    .buttonStyle(.borderedProminent)
                    .tint(MuseumPaperTheme.accent)
                    .controlSize(.large)
                    .accessibilityIdentifier("start-using-button")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
