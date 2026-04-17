import Foundation
import SemanticGalleryPersistence
import Testing
@testable import SemanticGalleryAppShell

@Test
func runtimeOptionsIgnoreEnvironmentOverridesWithoutOptInFlag() {
    let supportRoot = "/tmp/semanticgallery-runtime-support"
    let cachesRoot = "/tmp/semanticgallery-runtime-caches"
    let logsRoot = "/tmp/semanticgallery-runtime-logs"

    setenv("SEMANTICGALLERY_SUPPORT_ROOT", supportRoot, 1)
    setenv("SEMANTICGALLERY_CACHES_ROOT", cachesRoot, 1)
    setenv("SEMANTICGALLERY_LOGS_ROOT", logsRoot, 1)
    defer {
        unsetenv("SEMANTICGALLERY_SUPPORT_ROOT")
        unsetenv("SEMANTICGALLERY_CACHES_ROOT")
        unsetenv("SEMANTICGALLERY_LOGS_ROOT")
    }

    let options = SemanticGalleryRuntimeOptions.current(arguments: ["SemanticGallery"])
    let defaultSupportRoot = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appending(path: "SemanticGallery")

    #expect(options.paths.supportRoot.standardizedFileURL == defaultSupportRoot.standardizedFileURL)
    #expect(options.paths.supportRoot.standardizedFileURL != URL(filePath: supportRoot, directoryHint: .isDirectory).standardizedFileURL)
}

@Test
func runtimeOptionsKeepTheDefaultBundledArtifactsRootWhenOverridesAreDisabled() {
    let bundledArtifactsRoot = URL(
        filePath: "/tmp/semanticgallery-bundled-artifacts",
        directoryHint: .isDirectory
    )

    let options = SemanticGalleryRuntimeOptions.current(
        arguments: ["SemanticGallery"],
        defaultPaths: AppPaths(bundledArtifactsRoot: bundledArtifactsRoot)
    )

    #expect(options.paths.bundledArtifactsRoot?.standardizedFileURL == bundledArtifactsRoot.standardizedFileURL)
    #expect(options.paths.runtimeArtifactsRoot.standardizedFileURL == bundledArtifactsRoot.standardizedFileURL)
}

@Test
func runtimeOptionsPreferExplicitSupportCachesAndLogsRootsWhenOverridesAreEnabled() {
    let supportRoot = "/tmp/semanticgallery-runtime-support"
    let cachesRoot = "/tmp/semanticgallery-runtime-caches"
    let logsRoot = "/tmp/semanticgallery-runtime-logs"

    setenv("SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES", "1", 1)
    setenv("SEMANTICGALLERY_SUPPORT_ROOT", supportRoot, 1)
    setenv("SEMANTICGALLERY_CACHES_ROOT", cachesRoot, 1)
    setenv("SEMANTICGALLERY_LOGS_ROOT", logsRoot, 1)
    defer {
        unsetenv("SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES")
        unsetenv("SEMANTICGALLERY_SUPPORT_ROOT")
        unsetenv("SEMANTICGALLERY_CACHES_ROOT")
        unsetenv("SEMANTICGALLERY_LOGS_ROOT")
    }

    let options = SemanticGalleryRuntimeOptions.current(
        arguments: ["SemanticGallery", "--semanticgallery-runtime-overrides"]
    )

    #expect(options.paths.supportRoot.standardizedFileURL == URL(filePath: supportRoot, directoryHint: .isDirectory).standardizedFileURL)
    #expect(options.paths.cachesRoot.standardizedFileURL == URL(filePath: cachesRoot, directoryHint: .isDirectory).standardizedFileURL)
    #expect(options.paths.logsRoot.standardizedFileURL == URL(filePath: logsRoot, directoryHint: .isDirectory).standardizedFileURL)
}

@Test
func runtimeOptionsIgnoreOverrideEnvironmentWhenLaunchArgumentIsMissing() {
    let supportRoot = "/tmp/semanticgallery-runtime-support"

    setenv("SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES", "1", 1)
    setenv("SEMANTICGALLERY_SUPPORT_ROOT", supportRoot, 1)
    defer {
        unsetenv("SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES")
        unsetenv("SEMANTICGALLERY_SUPPORT_ROOT")
    }

    let options = SemanticGalleryRuntimeOptions.current(arguments: ["SemanticGallery"])
    let defaultSupportRoot = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appending(path: "SemanticGallery")

    #expect(options.paths.supportRoot.standardizedFileURL == defaultSupportRoot.standardizedFileURL)
    #expect(options.paths.supportRoot.standardizedFileURL != URL(filePath: supportRoot, directoryHint: .isDirectory).standardizedFileURL)
}

@Test
func runtimeDirectoryBootstrapperCreatesApplicationSupportSubdirectories() throws {
    let parentRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: parentRoot) }

    let paths = AppPaths(root: parentRoot)
    RuntimeDirectoryBootstrapper.ensureExists(for: paths)

    #expect(FileManager.default.fileExists(atPath: paths.supportRoot.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: paths.cachesRoot.path(percentEncoded: false)))
    #expect(FileManager.default.fileExists(atPath: paths.logsRoot.path(percentEncoded: false)))
}
