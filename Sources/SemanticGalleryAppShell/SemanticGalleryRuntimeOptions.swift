import Foundation
import SemanticGalleryPersistence

public struct SemanticGalleryRuntimeOptions: Sendable {
    public let paths: AppPaths
    public let defaultsSuiteName: String?
    public let useStubDownloads: Bool
    public let artifactSourceRoot: URL?
    public let installStepDelayNanoseconds: UInt64
    public let folderPreparationDelayNanoseconds: UInt64
    public let folderObservationDebounceNanoseconds: UInt64
    public let deferSelfUninstallTermination: Bool

    public init(
        paths: AppPaths,
        defaultsSuiteName: String?,
        useStubDownloads: Bool,
        artifactSourceRoot: URL? = nil,
        installStepDelayNanoseconds: UInt64,
        folderPreparationDelayNanoseconds: UInt64,
        folderObservationDebounceNanoseconds: UInt64 = 2_000_000_000,
        deferSelfUninstallTermination: Bool = false
    ) {
        self.paths = paths
        self.defaultsSuiteName = defaultsSuiteName
        self.useStubDownloads = useStubDownloads
        self.artifactSourceRoot = artifactSourceRoot
        self.installStepDelayNanoseconds = installStepDelayNanoseconds
        self.folderPreparationDelayNanoseconds = folderPreparationDelayNanoseconds
        self.folderObservationDebounceNanoseconds = folderObservationDebounceNanoseconds
        self.deferSelfUninstallTermination = deferSelfUninstallTermination
    }

    public static func current(
        processInfo: ProcessInfo = .processInfo,
        arguments: [String] = CommandLine.arguments,
        defaultPaths: @autoclosure () -> AppPaths = AppPaths()
    ) -> SemanticGalleryRuntimeOptions {
        let environment = processInfo.environment
        let runtimeOverridesEnabled =
            environment["SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES"] == "1"
            && arguments.contains("--semanticgallery-runtime-overrides")
        let bundledArtifactsRoot = runtimeOverridesEnabled
            ? (environment["SEMANTICGALLERY_BUNDLED_ARTIFACTS_ROOT"]
                ?? environment["SEMANTICGALLERY_ARTIFACT_SOURCE_ROOT"])
            : nil
        let bundledArtifactsURL: URL?
        if let bundledArtifactsRoot, bundledArtifactsRoot.isEmpty == false {
            bundledArtifactsURL = URL(fileURLWithPath: bundledArtifactsRoot, isDirectory: true)
        } else {
            bundledArtifactsURL = nil
        }
        let paths: AppPaths
        if runtimeOverridesEnabled,
           let supportRootPath = environment["SEMANTICGALLERY_SUPPORT_ROOT"],
           let cachesRootPath = environment["SEMANTICGALLERY_CACHES_ROOT"],
           let logsRootPath = environment["SEMANTICGALLERY_LOGS_ROOT"] {
            paths = AppPaths(
                supportRoot: URL(fileURLWithPath: supportRootPath, isDirectory: true),
                cachesRoot: URL(fileURLWithPath: cachesRootPath, isDirectory: true),
                logsRoot: URL(fileURLWithPath: logsRootPath, isDirectory: true),
                bundledArtifactsRoot: bundledArtifactsURL
            )
        } else if runtimeOverridesEnabled, let rootPath = environment["SEMANTICGALLERY_APP_ROOT"] {
            paths = AppPaths(
                root: URL(fileURLWithPath: rootPath, isDirectory: true),
                bundledArtifactsRoot: bundledArtifactsURL
            )
        } else {
            paths = defaultPaths()
        }

        return SemanticGalleryRuntimeOptions(
            paths: paths,
            defaultsSuiteName: runtimeOverridesEnabled ? environment["SEMANTICGALLERY_DEFAULTS_SUITE"] : nil,
            useStubDownloads: runtimeOverridesEnabled && environment["SEMANTICGALLERY_USE_STUB_DOWNLOADS"] == "1",
            artifactSourceRoot: runtimeOverridesEnabled ? environment["SEMANTICGALLERY_ARTIFACT_SOURCE_ROOT"].flatMap { value in
                guard value.isEmpty == false else {
                    return nil
                }
                return URL(fileURLWithPath: value, isDirectory: true)
            } : nil,
            installStepDelayNanoseconds: runtimeOverridesEnabled ? nanoseconds(fromMilliseconds: environment["SEMANTICGALLERY_INSTALL_STEP_DELAY_MS"]) : 0,
            folderPreparationDelayNanoseconds: runtimeOverridesEnabled ? nanoseconds(fromMilliseconds: environment["SEMANTICGALLERY_FOLDER_STEP_DELAY_MS"]) : 0,
            folderObservationDebounceNanoseconds: runtimeOverridesEnabled ? environment["SEMANTICGALLERY_FOLDER_OBSERVATION_DEBOUNCE_MS"].flatMap { value in
                let nanoseconds = nanoseconds(fromMilliseconds: value)
                return nanoseconds == 0 ? 2_000_000_000 : nanoseconds
            } ?? 2_000_000_000 : 2_000_000_000,
            deferSelfUninstallTermination: runtimeOverridesEnabled && environment["SEMANTICGALLERY_DEFER_SELF_UNINSTALL_TERMINATION"] == "1"
        )
    }

    public func makeDefaults() -> UserDefaults {
        if let defaultsSuiteName,
           let suiteDefaults = UserDefaults(suiteName: defaultsSuiteName) {
            return suiteDefaults
        }
        return .standard
    }

    private static func nanoseconds(fromMilliseconds value: String?) -> UInt64 {
        guard let value, let milliseconds = UInt64(value) else {
            return 0
        }
        return milliseconds * 1_000_000
    }
}
