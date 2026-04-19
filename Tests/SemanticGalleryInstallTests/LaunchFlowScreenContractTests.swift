import Testing
@testable import SemanticGalleryCore
@testable import SemanticGalleryInstall

@Test
func launchFlowScreenDependsOnlyOnTheCallerVisibleStatusClass() {
    /// Formal specification for callers:
    /// Precondition:
    ///   `status ∈ AppStatus`.
    /// Postcondition:
    ///   1. `status ∈ {installRequired, installFailed}`    ⇒ `LaunchFlowScreen(status) = .intro`
    ///   2. `status ∈ {installing, uninstalling}`          ⇒ `LaunchFlowScreen(status) = .progress`
    ///   3. `status = installComplete`                     ⇒ `LaunchFlowScreen(status) = .completion`
    ///   4. all workspace statuses                         ⇒ `LaunchFlowScreen(status) = .intro`
    #expect(LaunchFlowScreen(status: .installRequired) == .intro)
    #expect(LaunchFlowScreen(status: .installFailed("Failed")) == .intro)
    #expect(LaunchFlowScreen(status: .installing) == .progress)
    #expect(LaunchFlowScreen(status: .uninstalling) == .progress)
    #expect(LaunchFlowScreen(status: .installComplete) == .completion)
    #expect(LaunchFlowScreen(status: .readyWithoutFolder) == .intro)
    #expect(LaunchFlowScreen(status: .indexing) == .intro)
    #expect(LaunchFlowScreen(status: .searching) == .intro)
    #expect(LaunchFlowScreen(status: .training) == .intro)
    #expect(LaunchFlowScreen(status: .ready) == .intro)
}
