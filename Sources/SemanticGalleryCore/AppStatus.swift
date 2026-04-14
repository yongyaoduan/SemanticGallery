public enum AppStatus: Equatable, Sendable {
    case installRequired
    case installing
    case installFailed(String)
    case installComplete
    case readyWithoutFolder
    case preparingFolder
    case ready
    case uninstalling
}
