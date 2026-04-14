@MainActor
public struct AppContainer {
    public let statusStore: AppStatusStore
    public let libraryStateStore: LibraryStateStore

    public init(
        statusStore: AppStatusStore = AppStatusStore(),
        libraryStateStore: LibraryStateStore = LibraryStateStore()
    ) {
        self.statusStore = statusStore
        self.libraryStateStore = libraryStateStore
    }
}
