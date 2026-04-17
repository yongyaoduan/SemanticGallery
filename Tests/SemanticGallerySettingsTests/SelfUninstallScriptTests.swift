import Foundation
import Testing
@testable import SemanticGallerySettings

@Test
func selfRemovalScriptWaitsForTheCurrentProcessAndDeletesTheBundle() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }

    let appBundleURL = root.appending(path: "Applications").appending(path: "SemanticGallery.app")
    try FileManager.default.createDirectory(at: appBundleURL, withIntermediateDirectories: true)

    let coordinator = UninstallCoordinator()
    let scriptURL = try coordinator.writeSelfRemovalScript(
        appBundleURL: appBundleURL,
        processIdentifier: 4242,
        scriptRoot: root
    )
    let scriptContents = try String(contentsOf: scriptURL, encoding: .utf8)

    #expect(scriptContents.contains("kill -0 4242"))
    #expect(scriptContents.contains("sleep 3"))
    #expect(scriptContents.contains("rm -rf '\(appBundleURL.path(percentEncoded: false))'"))
}
