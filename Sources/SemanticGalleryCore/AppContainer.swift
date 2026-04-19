@MainActor
public struct AppContainer {
    public let statusState: StatusState
    public let libraryState: LibraryState

    public init(
        statusState: StatusState = StatusState(),
        libraryState: LibraryState = LibraryState()
    ) {
        self.statusState = statusState
        self.libraryState = libraryState
    }
}
