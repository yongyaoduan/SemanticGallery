import Observation

@MainActor
@Observable
public final class AppStatusStore {
    public var status: AppStatus

    public init(status: AppStatus = .readyWithoutFolder) {
        self.status = status
    }
}
