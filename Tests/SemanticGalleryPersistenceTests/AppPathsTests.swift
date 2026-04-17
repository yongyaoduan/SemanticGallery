import Foundation
import Testing
@testable import SemanticGalleryPersistence

@Test
func appPathsUseExplicitSupportCachesAndLogsRoots() {
    let supportRoot = URL(filePath: "/tmp/SemanticGallerySupport", directoryHint: .isDirectory)
    let cachesRoot = URL(filePath: "/tmp/SemanticGalleryCaches", directoryHint: .isDirectory)
    let logsRoot = URL(filePath: "/tmp/SemanticGalleryLogs", directoryHint: .isDirectory)

    let paths = AppPaths(
        supportRoot: supportRoot,
        cachesRoot: cachesRoot,
        logsRoot: logsRoot
    )

    #expect(paths.root == supportRoot)
    #expect(paths.supportRoot == supportRoot)
    #expect(paths.cachesRoot == cachesRoot)
    #expect(paths.logsRoot == logsRoot)
    #expect(paths.databaseURL == supportRoot.appending(path: "library.sqlite"))
    #expect(paths.installStateURL == supportRoot.appending(path: "install-state.json"))
    #expect(paths.artifactsRoot == supportRoot.appending(path: "Artifacts"))
}

@Test
func appPathsPreferBundledArtifactsWhenProvided() {
    let supportRoot = URL(filePath: "/tmp/SemanticGallerySupport", directoryHint: .isDirectory)
    let cachesRoot = URL(filePath: "/tmp/SemanticGalleryCaches", directoryHint: .isDirectory)
    let logsRoot = URL(filePath: "/tmp/SemanticGalleryLogs", directoryHint: .isDirectory)
    let bundledArtifactsRoot = URL(filePath: "/Applications/SemanticGallery.app/Contents/Resources/SemanticGalleryArtifacts", directoryHint: .isDirectory)

    let paths = AppPaths(
        supportRoot: supportRoot,
        cachesRoot: cachesRoot,
        logsRoot: logsRoot,
        bundledArtifactsRoot: bundledArtifactsRoot
    )

    #expect(paths.runtimeArtifactsRoot == bundledArtifactsRoot)
}

@Test
func appPathsDefaultWritableRootsStayInsideApplicationSupport() {
    let applicationSupportRoot = FileManager.default
        .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appending(path: "SemanticGallery")

    let paths = AppPaths(root: nil, bundledArtifactsRoot: nil)

    #expect(paths.root == applicationSupportRoot)
    #expect(paths.supportRoot == applicationSupportRoot)
    #expect(paths.cachesRoot == applicationSupportRoot.appending(path: "Caches"))
    #expect(paths.logsRoot == applicationSupportRoot.appending(path: "Logs"))
}
