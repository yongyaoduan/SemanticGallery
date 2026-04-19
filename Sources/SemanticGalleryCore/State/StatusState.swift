import Observation

@MainActor
@Observable
public final class StatusState {
    public var status: AppStatus

    public init(status: AppStatus = .readyWithoutFolder) {
        self.status = status
    }
}
