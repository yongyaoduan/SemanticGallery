import Testing
@testable import SemanticGalleryCore

@Test
@MainActor
func initialAppStatusRequiresInstall() {
    let store = AppStatusStore()
    #expect(store.status == .installRequired)
}
