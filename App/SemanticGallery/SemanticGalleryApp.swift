import AppKit
import SwiftUI
import SemanticGalleryAppShell

final class SemanticGalleryAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)

        for delay in [0.0, 0.15, 0.45] {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                application.unhide(nil)
                application.activate(ignoringOtherApps: true)
                application.windows.first?.makeKeyAndOrderFront(nil)
            }
        }
    }
}

@main
struct SemanticGalleryApp: App {
    @State private var controller: SemanticGalleryController
    @NSApplicationDelegateAdaptor(SemanticGalleryAppDelegate.self) private var appDelegate

    init() {
        let runtimeOptions = SemanticGalleryRuntimeOptions.current()
        RuntimeDirectoryBootstrapper.ensureExists(for: runtimeOptions.paths)
        _controller = State(initialValue: SemanticGalleryController(runtimeOptions: runtimeOptions))
    }

    var body: some Scene {
        WindowGroup {
            SemanticGalleryRootView(controller: controller)
                .frame(minWidth: 1200, minHeight: 780)
        }
        Settings {
            SemanticGallerySettingsScene(controller: controller)
        }
        .defaultSize(width: 760, height: 420)
    }
}
