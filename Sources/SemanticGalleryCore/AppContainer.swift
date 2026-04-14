@MainActor
public struct AppContainer {
    public let statusStore: AppStatusStore

    public init(statusStore: AppStatusStore = AppStatusStore()) {
        self.statusStore = statusStore
    }
}
