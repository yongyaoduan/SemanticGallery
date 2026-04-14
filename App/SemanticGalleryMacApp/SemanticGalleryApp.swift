import SwiftUI
import SemanticGalleryCore

@main
struct SemanticGalleryApp: App {
    @State private var container = AppContainer()

    var body: some Scene {
        WindowGroup {
            RootContentView(statusStore: container.statusStore)
                .frame(minWidth: 1200, minHeight: 780)
        }
    }
}
