import AppKit
import SwiftUI
import SemanticGalleryAppShell

final class SemanticGalleryAppDelegate: NSObject, NSApplicationDelegate {
    private let activationDelays: [TimeInterval] = [5.0]
    private var hasPendingWorkspaceWindowRequest = false

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(promoteObservedWindowToFront(_:)),
            name: NSWindow.didBecomeMainNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(promoteObservedWindowToFront(_:)),
            name: NSWindow.didBecomeKeyNotification,
            object: nil
        )

        for delay in activationDelays {
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                self?.openWorkspaceWindowIfNeeded()
                self?.promoteAppToFront()
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        NotificationCenter.default.removeObserver(self)
    }

    @MainActor
    @objc private func promoteObservedWindowToFront(_ notification: Notification) {
        promoteAppToFront(preferredWindow: notification.object as? NSWindow)
    }

    @MainActor
    private func promoteAppToFront(preferredWindow: NSWindow? = nil) {
        let application = NSApplication.shared
        application.unhide(nil)
        application.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [.activateAllWindows])
        if let window = frontmostWindow(in: application, preferredWindow: preferredWindow) {
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
        }
    }

    @MainActor
    private func openWorkspaceWindowIfNeeded() {
        let application = NSApplication.shared
        if application.windows.contains(where: { $0.isVisible || $0.isMiniaturized == false }) {
            hasPendingWorkspaceWindowRequest = false
            return
        }

        if hasPendingWorkspaceWindowRequest {
            return
        }

        guard
            let fileMenu = application.mainMenu?.items.first(where: { $0.title == "File" })?.submenu,
            let newWindowItem = fileMenu.items.first(where: { $0.title == "New Window" }),
            let action = newWindowItem.action
        else {
            return
        }

        hasPendingWorkspaceWindowRequest = true
        let didSendAction = application.sendAction(action, to: newWindowItem.target, from: newWindowItem)
        if didSendAction == false {
            hasPendingWorkspaceWindowRequest = false
        }
    }

    @MainActor
    private func frontmostWindow(
        in application: NSApplication,
        preferredWindow: NSWindow? = nil
    ) -> NSWindow? {
        if let preferredWindow,
           preferredWindow.isVisible {
            return preferredWindow
        }
        if let keyWindow = application.keyWindow,
           keyWindow.isVisible {
            return keyWindow
        }
        if let mainWindow = application.mainWindow,
           mainWindow.isVisible {
            return mainWindow
        }
        return application.windows.first(where: { $0.isVisible }) ?? application.windows.first
    }
}

@main
struct SemanticGalleryApp: App {
    @State private var controller: GalleryController
    @NSApplicationDelegateAdaptor(SemanticGalleryAppDelegate.self) private var appDelegate

    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        let runtimeOptions = RuntimeConfiguration.current()
        RuntimeDirectories.ensureExists(for: runtimeOptions.paths)
        _controller = State(initialValue: GalleryController(runtimeOptions: runtimeOptions))
    }

    var body: some Scene {
        WindowGroup {
            AppRootView(controller: controller)
                .frame(minWidth: 1200, minHeight: 780)
        }
        Settings {
            AppSettingsScene(controller: controller)
        }
        .defaultSize(width: 760, height: 420)
    }
}
