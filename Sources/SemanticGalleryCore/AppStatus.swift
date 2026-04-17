public enum AppStatus: Equatable, Sendable {
    case installRequired
    case installing
    case installFailed(String)
    case installComplete
    case readyWithoutFolder
    case indexing
    case searching
    case training
    case ready
    case uninstalling
}
