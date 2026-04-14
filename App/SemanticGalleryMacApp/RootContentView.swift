import SwiftUI
import SemanticGalleryCore

struct RootContentView: View {
    @Bindable var statusStore: AppStatusStore

    var body: some View {
        Group {
            switch statusStore.status {
            case .installRequired:
                Text("Install SemanticGallery")
            case .installing:
                Text("Installing SemanticGallery")
            case .installFailed(let message):
                Text(message)
            case .installComplete:
                Text("Installation Complete")
            case .readyWithoutFolder:
                Text("Choose Folder")
            case .preparingFolder:
                Text("Preparing Folder")
            case .ready:
                Text("Workspace")
            case .uninstalling:
                Text("Uninstalling")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
