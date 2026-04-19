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
        case .readyWithoutFolder, .indexing, .searching, .training, .ready:
            self = .intro
        }
    }
}

public struct LaunchFlowView: View {
    @Bindable private var statusState: StatusState
    @Bindable private var installationState: InstallationState
    private let startInstallation: () -> Void
    private let enterApp: () -> Void

    public init(
        statusState: StatusState,
        installationState: InstallationState,
        startInstallation: @escaping () -> Void,
        enterApp: @escaping () -> Void
    ) {
        self.statusState = statusState
        self.installationState = installationState
        self.startInstallation = startInstallation
        self.enterApp = enterApp
    }

    public var body: some View {
        ZStack {
            LinearGradient(
                colors: [MuseumPaperTheme.backgroundTop, MuseumPaperTheme.backgroundBottom],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            switch LaunchFlowScreen(status: statusState.status) {
            case .intro:
                LaunchIntroView(errorMessage: installFailureMessage, startInstallation: startInstallation)
            case .progress:
                LaunchProgressView(progress: installationState.progress)
            case .completion:
                LaunchCompletionView(enterApp: enterApp)
            }
        }
    }

    private var installFailureMessage: String? {
        guard case let .installFailed(message) = statusState.status else {
            return nil
        }
        return message
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
                .overlay(
                    RoundedRectangle(cornerRadius: 32, style: .continuous)
                        .stroke(MuseumPaperTheme.line, lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.08), radius: 28, y: 18)
        )
        .padding(48)
    }
}

private struct LaunchIntroView: View {
    let errorMessage: String?
    let startInstallation: () -> Void

    var body: some View {
        LaunchShell {
            VStack(alignment: .leading, spacing: 20) {
                Text("Prepare SemanticGallery on this Mac")
                    .font(.system(size: 42, weight: .semibold, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.ink)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("launch-intro-headline")

                VStack(alignment: .leading, spacing: 12) {
                    introInstallRow(title: "Lucas-tuned SigLIP2 encoder", detail: "For the first semantic index")
                    introInstallRow(title: "Shared configuration", detail: "Tokenizer, preprocessing, and metadata")
                    introInstallRow(title: "Reference image set", detail: "For the shared starting point")
                }
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("launch-intro-install-list")

                HStack(spacing: 14) {
                    MuseumActionButton(
                        "Start Installation",
                        accessibilityIdentifier: "start-installation-button",
                        action: startInstallation
                    )
                    .controlSize(.large)

                    Text("Nothing leaves this Mac after setup finishes.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MuseumPaperTheme.mutedInk)
                        .accessibilityIdentifier("launch-intro-privacy-note")
                }

                if let errorMessage {
                    Text(errorMessage)
                        .foregroundStyle(Color(red: 0.63, green: 0.26, blue: 0.20))
                        .accessibilityIdentifier("install-error-message")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func introInstallRow(title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(MuseumPaperTheme.accent)
                .frame(width: 8, height: 8)
                .padding(.top, 6)

            VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(MuseumPaperTheme.ink)
            Text(detail)
                .font(.system(size: 14))
                .foregroundStyle(MuseumPaperTheme.mutedInk)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct LaunchProgressView: View {
    let progress: [InstallProgress]

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { timeline in
            LaunchShell {
                VStack(alignment: .leading, spacing: 22) {
                    Text("Installing SemanticGallery")
                        .font(.system(size: 38, weight: .semibold, design: .serif))
                        .foregroundStyle(MuseumPaperTheme.ink)

                    Text("Preparing the local encoder, shared config, and reference data on this Mac.")
                        .foregroundStyle(MuseumPaperTheme.mutedInk)

                    progressSummaryCard(now: timeline.date)

                    LayeredProgressBar(progress: overallProgress(now: timeline.date), isActive: true)
                        .frame(height: 12)

                    VStack(spacing: 14) {
                        ForEach(InstallStep.allCases, id: \.self) { step in
                            let item = progress(for: step, now: timeline.date)
                            let display = liveDisplay(for: item, now: timeline.date)
                            progressRow(
                                step: step,
                                title: step.title,
                                progress: display.progress,
                                caption: item.message,
                                timing: timingText(for: item, now: timeline.date),
                                isActive: currentProgressItem(now: timeline.date).step == step
                            )
                        }
                    }
                }
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("launch-progress-screen")
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func progressRow(
        step: InstallStep,
        title: String,
        progress: Double,
        caption: String,
        timing: String?,
        isActive: Bool
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .foregroundStyle(MuseumPaperTheme.ink)
                Spacer()
                if let timing {
                    Text(timing)
                        .monospacedDigit()
                        .foregroundStyle(MuseumPaperTheme.mutedInk)
                }
            }
            Text(caption)
                .font(.system(size: 12))
                .foregroundStyle(MuseumPaperTheme.mutedInk)
                .frame(maxWidth: .infinity, alignment: .leading)
            LayeredProgressBar(progress: progress, isActive: isActive)
                .frame(height: 10)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(MuseumPaperTheme.backgroundTop.opacity(isActive || progress > 0 ? 0.48 : 0.28))
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("install-step-\(step.accessibilityKey)")
    }

    private func progressSummaryCard(now: Date) -> some View {
        let presentation = InstallProgressPresentation(progress: progress, now: now)
        let item = presentation.currentProgressItem
        return HStack(alignment: .top, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Current stage")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(MuseumPaperTheme.mutedInk)
                Text(item.step.title)
                    .font(.system(size: 24, weight: .semibold, design: .serif))
                    .foregroundStyle(MuseumPaperTheme.ink)
                Text(item.message)
                    .font(.system(size: 14))
                    .foregroundStyle(MuseumPaperTheme.mutedInk)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 16)

            VStack(alignment: .trailing, spacing: 8) {
                Text("\(Int((presentation.overallProgress * 100).rounded()))%")
                    .font(.system(size: 30, weight: .bold, design: .serif))
                    .monospacedDigit()
                    .foregroundStyle(MuseumPaperTheme.accentStrong)
                Text("\(presentation.currentStepIndex()) of \(InstallStep.allCases.count) sections in motion")
                    .font(.system(size: 13, weight: .medium))
                    .monospacedDigit()
                    .foregroundStyle(MuseumPaperTheme.mutedInk)
                if let timing = timingText(for: item, now: now) {
                    Text(timing)
                        .font(.system(size: 12))
                        .monospacedDigit()
                        .foregroundStyle(MuseumPaperTheme.mutedInk)
                }
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(MuseumPaperTheme.noteFill.opacity(0.72))
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .stroke(MuseumPaperTheme.line, lineWidth: 1)
                )
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("launch-progress-summary-card")
    }

    private func overallProgress(now: Date) -> Double {
        InstallProgressPresentation(progress: progress, now: now).overallProgress
    }

    private func currentProgressItem(now: Date) -> InstallProgress {
        InstallProgressPresentation(progress: progress, now: now).currentProgressItem
    }

    private func currentStepIndex(now: Date) -> Int {
        InstallProgressPresentation(progress: progress, now: now).currentStepIndex()
    }

    private func progress(for step: InstallStep, now: Date = .now) -> InstallProgress {
        InstallProgressPresentation(progress: progress, now: now).progress(for: step)
    }

    private func timingText(for item: InstallProgress, now: Date) -> String? {
        let display = liveDisplay(for: item, now: now)
        switch (display.elapsedSeconds, display.remainingSeconds) {
        case let (.some(elapsed), .some(remaining)):
            return "\(format(duration: elapsed)) elapsed · \(format(duration: remaining)) left"
        case let (.some(elapsed), nil):
            return "\(format(duration: elapsed)) elapsed"
        case let (nil, .some(remaining)):
            return "\(format(duration: remaining)) left"
        case (nil, nil):
            return nil
        }
    }

    private func format(duration: Int) -> String {
        let minutes = duration / 60
        let seconds = duration % 60
        if minutes == 0 {
            return "\(seconds)s"
        }
        return "\(minutes)m \(seconds)s"
    }

    private func liveDisplay(for item: InstallProgress, now: Date) -> LiveProgressDisplay {
        LiveProgressPresentation(
            progress: item.stepProgress,
            elapsedSeconds: item.elapsedSeconds,
            remainingSeconds: item.remainingSeconds,
            recordedAt: item.recordedAt
        )
        .displayed(at: now)
    }
}

private extension String {
    var accessibilitySlug: String {
        lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(of: ".", with: "")
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

                MuseumActionButton(
                    "Start Using SemanticGallery",
                    accessibilityIdentifier: "start-using-button",
                    action: enterApp
                )
                .controlSize(.large)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
