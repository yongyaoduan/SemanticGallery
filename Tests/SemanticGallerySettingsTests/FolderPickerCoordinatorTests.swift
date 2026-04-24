import Foundation
import Testing
@testable import SemanticGallerySettings

@Test
func folderPickerPrefersTheExplicitInitialDirectoryWhenItIsProvided() {
    let url = FolderPickerCoordinator.initialDirectoryURL(
        environment: [
            "SEMANTICGALLERY_FOLDER_PICKER_INITIAL_DIRECTORY": "/tmp/semanticgallery-open-panel-root",
            "SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT": "/tmp/semanticgallery-real-folder-tests",
        ]
    )

    #expect(url == URL(filePath: "/tmp/semanticgallery-open-panel-root", directoryHint: .isDirectory))
}

@Test
func folderPickerUsesTheUiTestFixtureRootWhenItIsProvided() {
    let url = FolderPickerCoordinator.initialDirectoryURL(
        environment: ["SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT": "/tmp/semanticgallery-real-folder-tests"]
    )

    #expect(url == URL(filePath: "/tmp/semanticgallery-real-folder-tests", directoryHint: .isDirectory))
}

@Test
func folderPickerDefaultsToThePicturesDirectoryInsteadOfATemporaryFixturePath() {
    let url = FolderPickerCoordinator.initialDirectoryURL(environment: [:])

    #expect(url.path(percentEncoded: false).contains("/tmp/semanticgallery-ui-fixtures") == false)
    #expect(url == FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask).first ?? FileManager.default.homeDirectoryForCurrentUser)
}
