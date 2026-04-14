import Testing
@testable import SemanticGalleryCore
@testable import SemanticGalleryInstall

@Test
func launchFlowUsesThreeSeparateScreens() {
    #expect(LaunchFlowScreen(status: .installRequired) == .intro)
    #expect(LaunchFlowScreen(status: .installing) == .progress)
    #expect(LaunchFlowScreen(status: .installComplete) == .completion)
}
