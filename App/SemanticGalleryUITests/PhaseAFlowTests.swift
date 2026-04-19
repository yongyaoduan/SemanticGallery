import AppKit
import XCTest
import SQLite3

@MainActor
final class PhaseAFlowTests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))
    }

    override func tearDownWithError() throws {
        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))
    }

    func testAppLaunchesDirectlyIntoWorkspaceWhenBundledArtifactsAreAvailable() throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "bundled-direct-launch")

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: false,
            artifactSourceRoot: try installedArtifactFixtureRoot().path(percentEncoded: false),
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertFalse(app.buttons["choose-folder-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        let pathBar = identifiedElement(in: app, identifier: "workspace-path-bar")
        let searchEditor = identifiedElement(in: app, identifier: "workspace-search-editor")
        let resultLimitPicker = identifiedElement(in: app, identifier: "workspace-result-limit-picker")
        let settingsButton = app.buttons["open-settings-button"]
        XCTAssertTrue(pathBar.waitForExistence(timeout: 5))
        XCTAssertTrue(searchEditor.waitForExistence(timeout: 5))
        XCTAssertTrue(resultLimitPicker.waitForExistence(timeout: 5))
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-empty-title").waitForExistence(timeout: 5))
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-empty-message").waitForExistence(timeout: 5))
        XCTAssertLessThan(pathBar.frame.height, 50)
        XCTAssertLessThan(abs(pathBar.frame.height - searchEditor.frame.height), 4)
        XCTAssertLessThan(abs(settingsButton.frame.width - settingsButton.frame.height), 3)
        XCTAssertLessThanOrEqual(settingsButton.frame.height, 46)
        saveScreenshot(named: "01-workspace-empty", in: app, evidenceRoot: evidenceRoot)
    }

    func testCompletedInstallStartsInWorkspaceWithoutFolderSelected() throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "completed-install-empty")

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["choose-folder-button"].exists)
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-path-bar").waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(waitForSettingsStatusTitle(in: app, toEqual: "No folder selected", timeout: 5))
        XCTAssertTrue(identifiedElement(in: app, identifier: "settings-library-empty-prompt").waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["Choose a library in Settings to open the archive."].exists)
        saveScreenshot(named: "workspace-empty-direct", in: app, evidenceRoot: evidenceRoot)
    }

    func testSettingsWindowStaysInteractiveAfterOpeningFromWorkspace() throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5), app.debugDescription)
        clickElement(app.buttons["open-settings-button"], in: app)

        let chooseFolderButton = app.buttons["settings-choose-folder-button"]
        XCTAssertTrue(chooseFolderButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(waitForSettingsStatusTitle(in: app, toEqual: "No folder selected", timeout: 5), app.debugDescription)

        XCTAssertTrue(
            waitForCondition(timeout: 7) {
                chooseFolderButton.exists && chooseFolderButton.isHittable
            },
            app.debugDescription
        )
    }

    func testWorkspaceWithoutFolderKeepsSearchAndPrimaryControlsInteractive() throws {
        let runtimeRoot = makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 5), app.debugDescription)
        enterText("still editable", into: searchInput, in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 5) {
                String(describing: self.searchInput(in: app).value ?? "").contains("still editable")
            },
            app.debugDescription
        )
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(app.staticTexts["No Selected Folder"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["Open Settings to choose a folder."].waitForExistence(timeout: 5))

        let selectionModeButton = app.buttons["workspace-selection-mode-button"]
        XCTAssertTrue(selectionModeButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(selectionModeButton.isEnabled)
        clickElement(selectionModeButton, in: app)
        XCTAssertTrue(app.staticTexts["0 selected"].waitForExistence(timeout: 5), app.debugDescription)

        clickElement(app.buttons["open-settings-button"], in: app)
        let startPrivateAdaptationButton = app.buttons["start-private-adaptation-button"]
        XCTAssertTrue(startPrivateAdaptationButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(startPrivateAdaptationButton.isEnabled)
        clickElement(startPrivateAdaptationButton, in: app)

        let adaptationAlert = app.sheets.firstMatch
        XCTAssertTrue(adaptationAlert.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(adaptationAlert.staticTexts["Private Album Adaptation"].waitForExistence(timeout: 5))
        XCTAssertTrue(
            adaptationAlert.staticTexts["Choose a folder in Settings before starting private adaptation."].waitForExistence(timeout: 5),
            adaptationAlert.debugDescription
        )
        clickElement(adaptationAlert.buttons["OK"], in: app)
    }

    func testSelectingFolderFromSettingsShowsBusyStatusAndReturnsToReady() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumPreparation")
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)

        let settingsChooseFolderButton = app.buttons["settings-choose-folder-button"]
        XCTAssertTrue(settingsChooseFolderButton.waitForExistence(timeout: 5))
        clickElement(settingsChooseFolderButton, in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        let selectedFolderText = app.staticTexts["settings-selected-folder"]
        XCTAssertTrue(selectedFolderText.waitForExistence(timeout: 5))
        XCTAssertTrue(
            waitForDisplayedText(of: selectedFolderText, toEqual: selectedFolder.path, timeout: 5),
            "Current label: \(selectedFolderText.label) value: \(String(describing: selectedFolderText.value))"
        )

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        let initialProgress = currentSettingsProgressPercentage(in: app) ?? 0
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                let statusTitle = self.displayedText(of: self.identifiedElement(in: app, identifier: "settings-status-title"))
                guard let currentProgress = self.currentSettingsProgressPercentage(in: app) else {
                    return statusTitle == "Ready"
                }
                return currentProgress > initialProgress || statusTitle == "Ready"
            }
        )
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
    }

    func testSelectingFolderFromSettingsTransitionsToUsageViewAndSearchWorks() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumUsageSearch")
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertFalse(app.buttons["choose-folder-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        let settingsChooseFolderButton = app.buttons["settings-choose-folder-button"]
        XCTAssertTrue(settingsChooseFolderButton.waitForExistence(timeout: 5))
        clickElement(settingsChooseFolderButton, in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)

        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10), app.debugDescription)
        let searchEditor = app.groups["workspace-search-editor"].firstMatch
        XCTAssertTrue(searchEditor.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-path-bar").waitForExistence(timeout: 5))
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-results-grid").waitForExistence(timeout: 5))
        XCTAssertEqual(app.buttons.matching(identifier: "workspace-result-cell").count, supportedImageCount(in: selectedFolder))
        XCTAssertGreaterThan(try embeddingCount(in: runtimeRoot), 0)
        XCTAssertGreaterThan(searchEditor.frame.width, app.windows.firstMatch.frame.width * 0.68)

        focusEditableElement(searchInput, in: app)
        enterText("sample-01", into: searchInput, in: app)
        submitSearch(using: searchInput, in: app)

        XCTAssertTrue(waitForResultCellCount(in: app, expected: 1, timeout: 5))
    }

    func testSelectionAndBatchDeleteRemoveFilesAndCleanTheIndex() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumDelete")
        let initialImageCount = supportedImageCount(in: selectedFolder)
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        let settingsChooseFolderButton = app.buttons["settings-choose-folder-button"]
        XCTAssertTrue(settingsChooseFolderButton.waitForExistence(timeout: 5))
        clickElement(settingsChooseFolderButton, in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: initialImageCount, timeout: 20))

        let selectionModeButton = app.buttons["workspace-selection-mode-button"]
        XCTAssertTrue(selectionModeButton.waitForExistence(timeout: 5))
        clickElement(selectionModeButton, in: app)
        XCTAssertTrue(app.staticTexts["0 selected"].waitForExistence(timeout: 5))

        let selectionToggleButton = app.buttons["workspace-selection-toggle-button"]
        XCTAssertTrue(selectionToggleButton.waitForExistence(timeout: 5))
        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(app.staticTexts["\(initialImageCount) selected"].waitForExistence(timeout: 5))

        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(app.staticTexts["0 selected"].waitForExistence(timeout: 5))

        let resultCells = app.buttons.matching(identifier: "workspace-result-cell")
        XCTAssertGreaterThanOrEqual(resultCells.count, 2)
        let firstRelativePath = resultCells.element(boundBy: 0).label
        let secondRelativePath = resultCells.element(boundBy: 1).label
        let firstAbsolutePath = selectedFolder.appending(path: firstRelativePath).path
        let secondAbsolutePath = selectedFolder.appending(path: secondRelativePath).path
        let firstDeletedImageURL = selectedFolder.appending(path: firstRelativePath)
        let firstDeletedImageData = try Data(contentsOf: firstDeletedImageURL)

        clickElement(resultCells.element(boundBy: 0), in: app)
        clickElement(resultCells.element(boundBy: 1), in: app)
        XCTAssertTrue(app.staticTexts["2 selected"].waitForExistence(timeout: 5))

        let deleteButton = app.buttons["workspace-delete-button"]
        XCTAssertTrue(deleteButton.waitForExistence(timeout: 5))
        clickElement(deleteButton, in: app)

        let moveToTrashButton = app.sheets.buttons["Move to Trash"]
        XCTAssertTrue(moveToTrashButton.waitForExistence(timeout: 5))
        clickElement(moveToTrashButton, in: app)

        XCTAssertTrue(waitForResultCellCount(in: app, expected: initialImageCount - 2, timeout: 10))
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstAbsolutePath))
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondAbsolutePath))
        XCTAssertEqual(try totalVisibleFileCount(in: runtimeRoot), initialImageCount - 2)
        XCTAssertEqual(try totalFileInstanceCount(in: runtimeRoot), initialImageCount - 2)
        XCTAssertEqual(try embeddingCount(in: runtimeRoot), initialImageCount - 2)

        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10))
        pasteImage(data: firstDeletedImageData, filename: firstDeletedImageURL.lastPathComponent, into: searchInput, in: app)
        XCTAssertTrue(app.staticTexts["Pasted image ready"].waitForExistence(timeout: 5))

        XCTAssertTrue(
            waitForCondition(timeout: 10) {
                let labels = self.visibleResultLabels(in: app)
                return labels.isEmpty == false
                    && labels.contains(firstRelativePath) == false
                    && labels.contains(secondRelativePath) == false
            }
        )
    }

    func testBrowseModeOpensPreviewAndSelectionModeChangesCellTapBehavior() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumUsageSearch")
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "workspace-preview")

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        relaunchIntoWorkspace(in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                app.buttons.matching(identifier: "workspace-result-cell").count >= 2
            },
            app.debugDescription
        )

        let resultCells = app.buttons.matching(identifier: "workspace-result-cell")
        XCTAssertGreaterThanOrEqual(resultCells.count, 2)
        let firstFilename = (resultCells.element(boundBy: 0).label as NSString).lastPathComponent
        let secondFilename = (resultCells.element(boundBy: 1).label as NSString).lastPathComponent
        clickElement(resultCells.element(boundBy: 0), in: app)

        let previewOverlay = identifiedElement(in: app, identifier: "workspace-preview-overlay")
        XCTAssertTrue(previewOverlay.waitForExistence(timeout: 5), app.debugDescription)
        let previewSimilarButton = app.buttons["workspace-preview-similar-button"]
        let previewInfoButton = app.buttons["workspace-preview-info-button"]
        let previewDeleteButton = app.buttons["workspace-preview-delete-button"]
        let previewCloseButton = app.buttons["workspace-preview-close-button"]
        let previewNextButton = app.buttons["workspace-preview-next-button"]
        let previewPreviousButton = app.buttons["workspace-preview-previous-button"]
        XCTAssertTrue(previewSimilarButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewInfoButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewDeleteButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewCloseButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewNextButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewPreviousButton.waitForExistence(timeout: 5), app.debugDescription)
        saveScreenshot(named: "01-preview", in: app, evidenceRoot: evidenceRoot)

        clickElement(previewInfoButton, in: app)
        let metadataPanel = identifiedElement(in: app, identifier: "workspace-preview-metadata")
        XCTAssertTrue(metadataPanel.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts[firstFilename].waitForExistence(timeout: 5), app.debugDescription)

        clickElement(previewNextButton, in: app)
        XCTAssertTrue(waitForNonExistence(of: metadataPanel, timeout: 5))
        clickElement(previewInfoButton, in: app)
        XCTAssertTrue(metadataPanel.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts[secondFilename].waitForExistence(timeout: 5), app.debugDescription)

        clickElement(previewPreviousButton, in: app)
        XCTAssertTrue(waitForNonExistence(of: metadataPanel, timeout: 5))
        clickElement(previewInfoButton, in: app)
        XCTAssertTrue(metadataPanel.waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts[firstFilename].waitForExistence(timeout: 5), app.debugDescription)
        saveScreenshot(named: "02-preview-metadata", in: app, evidenceRoot: evidenceRoot)

        clickElement(previewCloseButton, in: app)
        XCTAssertTrue(waitForNonExistence(of: previewOverlay, timeout: 5))

        let selectionModeButton = app.buttons["workspace-selection-mode-button"]
        XCTAssertTrue(selectionModeButton.waitForExistence(timeout: 5), app.debugDescription)
        clickElement(selectionModeButton, in: app)
        clickElement(resultCells.element(boundBy: 0), in: app)
        XCTAssertTrue(app.staticTexts["1 selected"].waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertFalse(previewOverlay.exists)

        let selectionToggleButton = app.buttons["workspace-selection-toggle-button"]
        XCTAssertTrue(selectionToggleButton.waitForExistence(timeout: 5), app.debugDescription)
        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 5) {
                app.staticTexts["\(supportedImageCount(in: selectedFolder)) selected"].exists
            },
            app.debugDescription
        )
        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 5) {
                app.staticTexts["0 selected"].exists
            },
            app.debugDescription
        )
    }

    func testPastingImageRunsSemanticSearchAgainstIndexedEmbeddings() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumImageSearch")
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let imageURLs = supportedImageURLs(in: selectedFolder)
        XCTAssertGreaterThanOrEqual(imageURLs.count, 1)
        let queryURL = imageURLs[0]
        let queryRelativePath = queryURL.lastPathComponent

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        let settingsChooseFolderButton = app.buttons["settings-choose-folder-button"]
        XCTAssertTrue(settingsChooseFolderButton.waitForExistence(timeout: 5))
        clickElement(settingsChooseFolderButton, in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        XCTAssertGreaterThan(try embeddingCount(in: runtimeRoot), 0)

        closeSettingsWindowIfPresent(in: app)

        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10))
        pasteImage(fileURL: queryURL, into: searchInput, in: app)
        XCTAssertTrue(app.staticTexts["Pasted image ready"].waitForExistence(timeout: 5))

        XCTAssertTrue(
            waitForCondition(timeout: 10) {
                self.firstVisibleResultLabel(in: app) == queryRelativePath
            },
            app.debugDescription
        )
    }

    func testNaturalLanguageSearchRanksYellowCatImagesAheadOfDistractors() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let semanticFixture = try semanticYellowCatFixture()
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "semantic-text-search")

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        let settingsChooseFolderButton = app.buttons["settings-choose-folder-button"]
        XCTAssertTrue(settingsChooseFolderButton.waitForExistence(timeout: 5))
        clickElement(settingsChooseFolderButton, in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: semanticFixture.folderURL)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        saveScreenshot(named: "01-semantic-library-ready", in: app, evidenceRoot: evidenceRoot)

        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10))
        let semanticSearchStart = Date()
        pasteText("a yellow tabby cat resting on the floor", into: searchInput, in: app)
        dismissTextSuggestions(in: app)
        submitSearch(using: searchInput, in: app)
        dismissTextSuggestions(in: app)

        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                let labels = Array(self.visibleResultLabels(in: app).prefix(3))
                return labels.count == 3 && Set(labels).isSubset(of: Set(semanticFixture.expectedTopResults))
            },
            app.debugDescription
        )
        let semanticSearchSeconds = Date().timeIntervalSince(semanticSearchStart)
        recordMetric(named: "semantic_text_search_seconds", value: semanticSearchSeconds, evidenceRoot: evidenceRoot)
        XCTAssertLessThan(semanticSearchSeconds, 6)
        saveScreenshot(named: "02-semantic-text-search", in: app, evidenceRoot: evidenceRoot)
    }

    func testInstalledDMGAppBundlesArtifactsAndIndexesFromSettings() throws {
        let selectedFolder = try fixtureFolder(named: "AlbumPreparation")
        let evidenceRoot = makeEvidenceDirectory(named: "installed-dmg-real-flow")
        let distributionAppURL = try mountedDMGAppURL()
        let runtimeRoot = makeTemporaryDirectory()
        let suiteName = "SemanticGalleryUITests.InstalledDMG.\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))

        let app = launchApplication(
            at: distributionAppURL,
            suiteName: suiteName,
            runtimeRoot: runtimeRoot
        )
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            distributionAppURL.standardizedFileURL
        )

        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 20), app.debugDescription)
        assertBundledArtifactsAvailable(at: distributionAppURL)
        saveScreenshot(named: "01-installed-workspace-empty", in: app, evidenceRoot: evidenceRoot)

        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 10))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsPreparationToStartOrFinish(in: app, timeout: 10), app.debugDescription)
        saveScreenshot(named: "02a-installed-after-folder-choice", in: app, evidenceRoot: evidenceRoot)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: supportedImageCount(in: selectedFolder), timeout: 30))
        saveScreenshot(named: "03-installed-workspace", in: app, evidenceRoot: evidenceRoot)
    }

    func testInstalledDMGAppIndexesAndSearchesARealAlbumFolderWithoutTemporaryRuntimeRoots() throws {
        let selectedFolder = try realWorldAlbumFolder()
        let supportedImages = supportedImageURLs(in: selectedFolder)
        let distributionAppURL = try mountedDMGAppURL()
        let runtimeRoot = makeTemporaryDirectory()
        let suiteName = "SemanticGalleryUITests.InstalledDMG.\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }
        guard supportedImages.count >= 2 else {
            XCTFail("The real-world album folder needs at least two supported images.")
            return
        }

        let textQueryImage = supportedImages[0]
        let imageQueryImage = supportedImages[1]
        let expectedImageCount = supportedImages.count

        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))

        let app = launchApplication(
            at: distributionAppURL,
            suiteName: suiteName,
            runtimeRoot: runtimeRoot
        )
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            distributionAppURL.standardizedFileURL
        )

        assertBundledArtifactsAvailable(at: distributionAppURL)
        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 20), app.debugDescription)
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 10))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        let databaseURL = appSupportRoot(in: runtimeRoot).appending(path: "library.sqlite")
        let logURL = logsRoot(in: runtimeRoot).appending(path: "semanticgallery.log")

        XCTAssertTrue(
            waitForCondition(timeout: 180) {
                (try? self.scalarCount(in: databaseURL, sql: "SELECT COUNT(*) FROM file_instances")) == expectedImageCount
            },
            "The selected folder was not fully indexed in the installed app database."
        )
        XCTAssertTrue(
            waitForCondition(timeout: 180) {
                (try? self.embeddingCount(in: runtimeRoot, encoderVersion: "stage1")) == expectedImageCount
            },
            "The installed app did not write embeddings for the selected folder. Recent log lines:\n\(recentLogLines(at: logURL))"
        )
        let expectedTitle = expectedImageCount == 1
            ? "1 image ready"
            : "\(expectedImageCount) images ready"
        let statusTitle = identifiedElement(in: app, identifier: "settings-status-title")
        if statusTitle.exists {
            XCTAssertTrue(
                waitForDisplayedText(
                    of: statusTitle,
                    toEqual: expectedTitle,
                    timeout: 20
                ),
                app.debugDescription
            )
        }

        closeSettingsWindowIfPresent(in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: expectedImageCount, timeout: 180), app.debugDescription)
        XCTAssertEqual(
            try embeddingCount(in: runtimeRoot, encoderVersion: "stage1"),
            expectedImageCount
        )

        let logText = try String(
            contentsOf: logURL,
            encoding: .utf8
        )
        XCTAssertTrue(logText.contains("Preparing the semantic search model from the app bundle."))
        XCTAssertTrue(logText.contains("Semantic search model is ready."))
        XCTAssertTrue(logText.contains("Folder preparation finished."))
        XCTAssertTrue(logText.contains(selectedFolder.path(percentEncoded: false)))

        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10), app.debugDescription)

        clearText(in: searchInput, in: app)
        enterText(textQueryImage.deletingPathExtension().lastPathComponent, into: searchInput, in: app)
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                self.visibleResultLabels(in: app).contains(textQueryImage.lastPathComponent)
            },
            app.debugDescription
        )

        clearText(in: searchInput, in: app)
        pasteImage(fileURL: imageQueryImage, into: searchInput, in: app)
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                self.firstVisibleResultLabel(in: app) == imageQueryImage.lastPathComponent
            },
            app.debugDescription
        )
    }

    func testInstalledDMGAppReturnsToWorkspaceAfterReinstall() throws {
        let evidenceRoot = makeEvidenceDirectory(named: "installed-dmg-reinstall-workspace")
        let distributionAppURL = try mountedDMGAppURL()
        let runtimeRoot = makeTemporaryDirectory()
        let suiteName = "SemanticGalleryUITests.InstalledDMG.\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))

        let app = launchApplication(
            at: distributionAppURL,
            suiteName: suiteName,
            runtimeRoot: runtimeRoot
        )
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            distributionAppURL.standardizedFileURL
        )

        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertTrue(app.staticTexts["No Selected Folder"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Open Settings to choose a folder."].waitForExistence(timeout: 10))
        saveScreenshot(named: "01-installed-relaunch-workspace", in: app, evidenceRoot: evidenceRoot)
        app.terminate()
    }

    func testPrivateAdaptationRequiresOneHundredImagesBeforeTrainingBegins() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumPrivateAdaptationMinimum")
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "private-adaptation-minimum")

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        clickElement(app.buttons["open-settings-button"], in: app)
        let startPrivateAdaptationButton = app.buttons["start-private-adaptation-button"]
        XCTAssertTrue(startPrivateAdaptationButton.waitForExistence(timeout: 10))
        XCTAssertTrue(startPrivateAdaptationButton.isEnabled)
        clickElement(startPrivateAdaptationButton, in: app)

        let minimumAlert = app.sheets.firstMatch
        XCTAssertTrue(minimumAlert.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertTrue(minimumAlert.staticTexts["At least 100 images are needed"].waitForExistence(timeout: 5))
        XCTAssertGreaterThanOrEqual(minimumAlert.staticTexts.count, 2, minimumAlert.debugDescription)
        XCTAssertFalse(identifiedElement(in: app, identifier: "settings-status-activity").exists)
        clickElement(minimumAlert.buttons["OK"], in: app)
        saveScreenshot(named: "01-minimum-notice", in: app, evidenceRoot: evidenceRoot)
    }

    func testPrivateAdaptationRunsRealTrainingRebuildsTheIndexAndProducesArtifacts() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "PrivateAlbumTraining")
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "private-adaptation")

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 300), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        XCTAssertEqual(try embeddingCount(in: runtimeRoot), supportedImageCount(in: selectedFolder))

        clickElement(app.buttons["open-settings-button"], in: app)
        let startPrivateAdaptationButton = app.buttons["start-private-adaptation-button"]
        XCTAssertTrue(startPrivateAdaptationButton.waitForExistence(timeout: 30))
        saveScreenshot(named: "01-ready-for-training", in: app, evidenceRoot: evidenceRoot)

        let trainingStart = Date()
        clickElement(startPrivateAdaptationButton, in: app)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Adapting", timeout: 10), app.debugDescription)
        saveScreenshot(named: "02-training-progress", in: app, evidenceRoot: evidenceRoot)

        XCTAssertTrue(
            waitForCondition(timeout: 1800) {
                guard let latestTrainingRun = try? self.latestTrainingRun(in: runtimeRoot) else {
                    return false
                }
                guard latestTrainingRun.status == "completed" else {
                    return false
                }
                let stage2EmbeddingCount = (try? self.embeddingCount(in: runtimeRoot, encoderVersion: latestTrainingRun.encoderVersion)) ?? 0
                return stage2EmbeddingCount == self.supportedImageCount(in: selectedFolder)
                    && self.adaptedWeightFiles(in: runtimeRoot).isEmpty == false
                    && self.adaptedSummaryFiles(in: runtimeRoot).isEmpty == false
            },
            app.debugDescription
        )
        recordMetric(
            named: "private_adaptation_seconds",
            value: Date().timeIntervalSince(trainingStart),
            evidenceRoot: evidenceRoot,
            details: ["folder": selectedFolder.lastPathComponent]
        )
        XCTAssertLessThan(Date().timeIntervalSince(trainingStart), 90)

        let latestTrainingRun = try XCTUnwrap(try latestTrainingRun(in: runtimeRoot))
        XCTAssertEqual(latestTrainingRun.status, "completed")
        XCTAssertEqual(
            try embeddingCount(in: runtimeRoot, encoderVersion: latestTrainingRun.encoderVersion),
            supportedImageCount(in: selectedFolder)
        )

        closeSettingsWindowIfPresent(in: app)
        let searchInput = searchInput(in: app)
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10))

        focusEditableElement(searchInput, in: app)
        enterText("private-001", into: searchInput, in: app)
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: 1, timeout: 20))

        clearText(in: searchInput, in: app)
        let queryURL = try XCTUnwrap(supportedImageURLs(in: selectedFolder).first)
        pasteImage(fileURL: queryURL, into: searchInput, in: app)
        XCTAssertTrue(app.staticTexts["Pasted image ready"].waitForExistence(timeout: 5))
        XCTAssertTrue(
            waitForCondition(timeout: 30) {
                self.firstVisibleResultLabel(in: app) == queryURL.lastPathComponent
            },
            app.debugDescription
        )
        saveScreenshot(named: "03-training-complete", in: app, evidenceRoot: evidenceRoot)
    }

    func testWorkspaceValidationCapturesSearchDeleteThumbnailAndSelectionEvidence() throws {
        let runtimeRoot = makeTemporaryDirectory()
        let selectedFolder = try fixtureFolder(named: "AlbumWorkspaceValidation")
        let initialImageCount = supportedImageCount(in: selectedFolder)
        defer { try? FileManager.default.removeItem(at: runtimeRoot) }
        let evidenceRoot = makeEvidenceDirectory(named: "workspace-validation")

        try createInstalledRuntime(at: runtimeRoot)

        let suiteName = "SemanticGalleryUITests.\(UUID().uuidString)"
        defer { UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName) }

        let app = configuredApp(
            runtimeRoot: runtimeRoot,
            suiteName: suiteName,
            useStubDownloads: true,
            artifactSourceRoot: nil,
            installStepDelayMilliseconds: 0,
            folderPreparationDelayMilliseconds: 900
        )

        launchConfiguredApplication(app)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)

        let indexingStart = Date()
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)
        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        saveScreenshot(named: "01-folder-progress", in: app, evidenceRoot: evidenceRoot)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 300), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        recordMetric(
            named: "indexing_seconds",
            value: Date().timeIntervalSince(indexingStart),
            evidenceRoot: evidenceRoot,
            details: ["folder": selectedFolder.lastPathComponent]
        )
        XCTAssertLessThan(Date().timeIntervalSince(indexingStart), 18)

        closeSettingsWindowIfPresent(in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: initialImageCount, timeout: 20))
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-results-grid").waitForExistence(timeout: 5))
        saveScreenshot(named: "02-workspace-grid", in: app, evidenceRoot: evidenceRoot)

        let thumbnailStart = Date()
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                self.thumbnailLoadedCount(in: app) >= min(5, initialImageCount)
            },
            app.debugDescription
        )
        let thumbnailSeconds = Date().timeIntervalSince(thumbnailStart)
        recordMetric(named: "thumbnail_first_five_seconds", value: thumbnailSeconds, evidenceRoot: evidenceRoot)
        XCTAssertLessThan(thumbnailSeconds, 2)

        let searchInput = searchInput(in: app)
        let textQueryURL = try XCTUnwrap(supportedImageURLs(in: selectedFolder).first)
        let textQuery = textQueryURL.deletingPathExtension().lastPathComponent
        XCTAssertTrue(searchInput.waitForExistence(timeout: 10))
        let textSearchStart = Date()
        focusEditableElement(searchInput, in: app)
        enterText(textQuery, into: searchInput, in: app)
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                let labels = self.hittableResultLabels(in: app)
                return labels.count == 1 && labels[0].contains(textQueryURL.lastPathComponent)
            },
            app.debugDescription
        )
        let textSearchSeconds = Date().timeIntervalSince(textSearchStart)
        recordMetric(named: "text_search_seconds", value: textSearchSeconds, evidenceRoot: evidenceRoot)
        XCTAssertLessThan(textSearchSeconds, 5)
        saveScreenshot(named: "03-text-search", in: app, evidenceRoot: evidenceRoot)

        clearText(in: searchInput, in: app)
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: initialImageCount, timeout: 20))

        let queryURL = textQueryURL
        let imageSearchStart = Date()
        pasteImage(fileURL: queryURL, into: searchInput, in: app)
        XCTAssertTrue(app.staticTexts["Pasted image ready"].waitForExistence(timeout: 5))
        XCTAssertTrue(
            waitForCondition(timeout: 20) {
                self.firstVisibleResultLabel(in: app) == queryURL.lastPathComponent
            },
            app.debugDescription
        )
        let imageSearchSeconds = Date().timeIntervalSince(imageSearchStart)
        recordMetric(named: "image_search_seconds", value: imageSearchSeconds, evidenceRoot: evidenceRoot)
        XCTAssertLessThan(imageSearchSeconds, 6)

        clearText(in: searchInput, in: app)
        if app.buttons["Remove Image"].exists {
            clickElement(app.buttons["Remove Image"], in: app)
        }
        submitSearch(using: searchInput, in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: initialImageCount, timeout: 20))

        let selectionModeButton = app.buttons["workspace-selection-mode-button"]
        XCTAssertTrue(selectionModeButton.waitForExistence(timeout: 5))
        clickElement(selectionModeButton, in: app)

        let selectionToggleButton = app.buttons["workspace-selection-toggle-button"]
        XCTAssertTrue(selectionToggleButton.waitForExistence(timeout: 5))
        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(app.staticTexts["\(initialImageCount) selected"].waitForExistence(timeout: 5))
        saveScreenshot(named: "04-selection", in: app, evidenceRoot: evidenceRoot)
        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(app.staticTexts["0 selected"].waitForExistence(timeout: 5))

        let resultCells = app.buttons.matching(identifier: "workspace-result-cell")
        XCTAssertGreaterThanOrEqual(resultCells.count, 2)
        let firstRelativePath = resultCells.element(boundBy: 0).label
        let secondRelativePath = resultCells.element(boundBy: 1).label
        let firstAbsolutePath = selectedFolder.appending(path: firstRelativePath).path
        let secondAbsolutePath = selectedFolder.appending(path: secondRelativePath).path

        clickElement(resultCells.element(boundBy: 0), in: app)
        clickElement(resultCells.element(boundBy: 1), in: app)
        XCTAssertTrue(app.staticTexts["2 selected"].waitForExistence(timeout: 5))

        let deleteStart = Date()
        clickElement(app.buttons["workspace-delete-button"], in: app)
        let moveToTrashButton = app.sheets.buttons["Move to Trash"]
        XCTAssertTrue(moveToTrashButton.waitForExistence(timeout: 5))
        clickElement(moveToTrashButton, in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: initialImageCount - 2, timeout: 20))
        let deleteSeconds = Date().timeIntervalSince(deleteStart)
        recordMetric(named: "delete_seconds", value: deleteSeconds, evidenceRoot: evidenceRoot)
        XCTAssertLessThan(deleteSeconds, 6)

        XCTAssertFalse(FileManager.default.fileExists(atPath: firstAbsolutePath))
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondAbsolutePath))
        XCTAssertEqual(try totalVisibleFileCount(in: runtimeRoot), initialImageCount - 2)
        XCTAssertEqual(try totalFileInstanceCount(in: runtimeRoot), initialImageCount - 2)
        XCTAssertEqual(try embeddingCount(in: runtimeRoot), initialImageCount - 2)
        saveScreenshot(named: "05-after-delete", in: app, evidenceRoot: evidenceRoot)
    }

    private func configuredApp(
        runtimeRoot: URL,
        suiteName: String,
        useStubDownloads: Bool,
        artifactSourceRoot: String?,
        installStepDelayMilliseconds: Int,
        folderPreparationDelayMilliseconds: Int,
        deferSelfUninstallTermination: Bool = true
    ) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments.append("--semanticgallery-runtime-overrides")
        app.launchEnvironment = [
            "SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES": "1",
            "SEMANTICGALLERY_APP_ROOT": runtimeRoot.path,
            "SEMANTICGALLERY_DEFAULTS_SUITE": suiteName,
            "SEMANTICGALLERY_USE_STUB_DOWNLOADS": useStubDownloads ? "1" : "0",
            "SEMANTICGALLERY_INSTALL_STEP_DELAY_MS": "\(installStepDelayMilliseconds)",
            "SEMANTICGALLERY_FOLDER_STEP_DELAY_MS": "\(folderPreparationDelayMilliseconds)",
            "SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT": fixtureRoot().path,
            "ApplePersistenceIgnoreState": "YES",
        ]
        if deferSelfUninstallTermination {
            app.launchEnvironment["SEMANTICGALLERY_DEFER_SELF_UNINSTALL_TERMINATION"] = "1"
        }
        if let artifactSourceRoot {
            app.launchEnvironment["SEMANTICGALLERY_ARTIFACT_SOURCE_ROOT"] = artifactSourceRoot
        }
        return app
    }

    private func attachedInstalledApp() -> XCUIApplication {
        XCUIApplication()
    }

    private func launchConfiguredApplication(_ app: XCUIApplication) {
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 20))

        activateSemanticGalleryIfNeeded()
        if waitForApplicationSurface(in: app, timeout: 5) == false {
            requestWorkspaceWindow(in: app)
        }

        XCTAssertTrue(waitForApplicationSurface(in: app, timeout: 15), app.debugDescription)
    }

    private func launchApplication(
        at appURL: URL,
        suiteName: String,
        runtimeRoot: URL? = nil
    ) -> XCUIApplication {
        let app = attachedInstalledApp()
        app.launchArguments += [
            "--semanticgallery-runtime-overrides",
            "-ApplePersistenceIgnoreState", "YES",
            "-NSQuitAlwaysKeepsWindows", "NO",
        ]
        app.launchEnvironment["SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES"] = "1"
        app.launchEnvironment["SEMANTICGALLERY_DEFAULTS_SUITE"] = suiteName
        if let runtimeRoot {
            app.launchEnvironment["SEMANTICGALLERY_APP_ROOT"] = runtimeRoot.path(percentEncoded: false)
            app.launchEnvironment["SEMANTICGALLERY_BUNDLED_ARTIFACTS_ROOT"] =
                appURL
                .appending(path: "Contents")
                .appending(path: "Resources")
                .appending(path: "SemanticGalleryArtifacts")
                .path(percentEncoded: false)
        }
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 20), app.debugDescription)
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            appURL.standardizedFileURL
        )

        activateSemanticGalleryIfNeeded()

        if waitForApplicationSurface(in: app, timeout: 5) == false {
            requestWorkspaceWindow(in: app)
        }

        XCTAssertTrue(waitForApplicationSurface(in: app, timeout: 15), app.debugDescription)
        XCTAssertTrue(waitForWindowCount(in: app, expected: 1, timeout: 10), app.debugDescription)
        return app
    }

    private func createInstalledRuntime(at runtimeRoot: URL) throws {
        let supportRoot = appSupportRoot(in: runtimeRoot)
        try FileManager.default.createDirectory(at: supportRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: cachesRoot(in: runtimeRoot), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: logsRoot(in: runtimeRoot), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: supportRoot.appending(path: "library.sqlite").path, contents: Data())
        try Data(#"{"status":"complete"}"#.utf8).write(to: supportRoot.appending(path: "install-state.json"))
        try copyTree(
            from: try installedArtifactFixtureRoot(),
            to: supportRoot.appending(path: "Artifacts")
        )
    }

    private func waitForNonExistence(of element: XCUIElement, timeout: TimeInterval) -> Bool {
        let predicate = NSPredicate(format: "exists == false")
        let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
        return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
    }

    private func waitForDisplayedText(of element: XCUIElement, toEqual expectedText: String, timeout: TimeInterval) -> Bool {
        let normalizedExpected = normalizePathLikeText(expectedText)
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if let displayedText = displayedText(of: element), normalizePathLikeText(displayedText) == normalizedExpected {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return false
    }

    private func waitForSettingsStatusTitle(
        in app: XCUIApplication,
        toEqual expectedText: String,
        timeout: TimeInterval
    ) -> Bool {
        waitForDisplayedText(
            of: identifiedElement(in: app, identifier: "settings-status-title"),
            toEqual: expectedText,
            timeout: timeout
        )
    }

    private func waitForSettingsBusyState(
        in app: XCUIApplication,
        title: String,
        timeout: TimeInterval
    ) -> Bool {
        waitForCondition(timeout: timeout) {
            self.displayedText(of: self.identifiedElement(in: app, identifier: "settings-status-title")) == title
                && self.identifiedElement(in: app, identifier: "settings-status-activity").exists
                && self.currentSettingsProgressPercentage(in: app) != nil
        }
    }

    private func waitForSettingsReadyState(in app: XCUIApplication, timeout: TimeInterval) -> Bool {
        waitForCondition(timeout: timeout) {
            guard let title = self.displayedText(of: self.identifiedElement(in: app, identifier: "settings-status-title")) else {
                return false
            }
            return title.isEmpty == false
                && title != "Indexing"
                && title != "Adapting"
                && self.identifiedElement(in: app, identifier: "settings-status-activity").exists == false
        }
    }

    private func waitForSettingsPreparationToStartOrFinish(in app: XCUIApplication, timeout: TimeInterval) -> Bool {
        waitForCondition(timeout: timeout) {
            let title = self.displayedText(of: self.identifiedElement(in: app, identifier: "settings-status-title"))
            let isBusy = title == "Indexing"
                && self.identifiedElement(in: app, identifier: "settings-status-activity").exists
                && self.currentSettingsProgressPercentage(in: app) != nil
            let isReady = {
                guard let title else {
                    return false
                }
                return title.isEmpty == false
                    && title != "Indexing"
                    && title != "Adapting"
                    && self.identifiedElement(in: app, identifier: "settings-status-activity").exists == false
            }()
            return isBusy || isReady
        }
    }

    private func currentSettingsProgressPercentage(in app: XCUIApplication) -> Int? {
        let progressLabel = identifiedElement(in: app, identifier: "settings-status-progress")
        guard progressLabel.exists, let text = displayedText(of: progressLabel) else {
            return nil
        }

        let cleaned = text.replacingOccurrences(of: "%", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        return Int(cleaned)
    }

    private func chooseFolderThroughOpenPanel(in app: XCUIApplication, folderURL: URL) {
        guard let openPanel = waitForOpenPanel(in: app, timeout: 5) else {
            XCTFail("The folder chooser did not appear.")
            return
        }

        let navigatedDirectly = navigateOpenPanel(to: folderURL, in: openPanel, app: app)
        if navigatedDirectly == false {
            guard let folderItem = waitForFolderItem(named: folderURL.lastPathComponent, in: openPanel.panel, timeout: 2) else {
                XCTFail("The requested folder did not appear in the chooser.\n\(openPanel.panel.debugDescription)")
                return
            }
            clickElement(folderItem)
            RunLoop.current.run(until: Date().addingTimeInterval(0.2))
        }

        guard let chooseButton = waitForChooseButton(in: openPanel.panel, ownerApp: openPanel.ownerApp, timeout: 5) else {
            XCTFail("The confirmation button did not appear.")
            return
        }
        clickElement(chooseButton, in: app)

        XCTAssertTrue(waitForNonExistence(of: openPanel.panel, timeout: 5))
    }

    private func waitForOpenPanel(in app: XCUIApplication, timeout: TimeInterval) -> OpenPanelContext? {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            let appCandidates = [app.sheets.firstMatch, app.dialogs.firstMatch, app.windows.firstMatch]
            if let panel = appCandidates.first(where: { $0.exists }) {
                return OpenPanelContext(ownerApp: app, panel: panel)
            }

            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return nil
    }

    private func navigateOpenPanel(to folderURL: URL, in context: OpenPanelContext, app: XCUIApplication) -> Bool {
        context.ownerApp.activate()
        context.ownerApp.typeKey("G", modifierFlags: [.command, .shift])

        let deadline = Date().addingTimeInterval(5)
        repeat {
            let candidates = [
                context.ownerApp.sheets.firstMatch,
                context.ownerApp.dialogs.firstMatch,
                context.panel.sheets.firstMatch,
            ]
            if let goToPanel = candidates.first(where: { $0.exists }) {
                let textField = goToPanel.textFields.firstMatch.exists
                    ? goToPanel.textFields.firstMatch
                    : goToPanel.comboBoxes.textFields.firstMatch
                guard textField.exists else {
                    return false
                }
                clearText(in: textField, in: app)
                pasteText(folderURL.path(percentEncoded: false), into: textField, in: app)

                let buttons = [
                    goToPanel.buttons["Go"],
                    goToPanel.buttons["Open"],
                ]
                if let button = buttons.first(where: { $0.exists }) {
                    clickElement(button, in: app)
                } else {
                    context.ownerApp.typeKey(XCUIKeyboardKey.return.rawValue, modifierFlags: [])
                }

                RunLoop.current.run(until: Date().addingTimeInterval(0.4))
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return false
    }

    private func waitForFolderItem(named name: String, in panel: XCUIElement, timeout: TimeInterval) -> XCUIElement? {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            let listView = panel.outlines["ListView"]
            if listView.exists {
                let rows = listView.children(matching: .outlineRow)
                let visibleCount = rows.count
                if visibleCount > 0 {
                    for index in 0..<visibleCount {
                        let row = rows.element(boundBy: index)
                        let titleFields = row.descendants(matching: .textField)
                        if titleFields.count > 0 {
                            let value = titleFields.element(boundBy: 0).value as? String
                            if value == name {
                                return row
                            }
                        }
                    }
                }
            }

            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return nil
    }

    private func waitForChooseButton(
        in panel: XCUIElement,
        ownerApp: XCUIApplication,
        timeout: TimeInterval
    ) -> XCUIElement? {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            let candidates = [
                panel.buttons["Choose Folder"],
                panel.buttons["Choose"],
                panel.buttons["Open"],
                ownerApp.sheets.buttons["Choose Folder"],
                ownerApp.sheets.buttons["Choose"],
                ownerApp.sheets.buttons["Open"],
                ownerApp.dialogs.buttons["Choose Folder"],
                ownerApp.dialogs.buttons["Choose"],
                ownerApp.dialogs.buttons["Open"],
            ]
            if let button = candidates.first(where: { $0.exists }) {
                return button
            }

            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return nil
    }

    private func closeSettingsWindowIfPresent(in app: XCUIApplication) {
        let settingsWindowPredicate = NSPredicate(
            format: "identifier == %@ OR title CONTAINS %@",
            "com_apple_SwiftUI_Settings_window",
            "Settings"
        )
        let settingsWindows = app.windows.matching(settingsWindowPredicate)
        guard settingsWindows.count > 0 else {
            if waitForApplicationSurface(in: app, timeout: 1) == false {
                requestWorkspaceWindow(in: app)
            }
            return
        }

        relaunchIntoWorkspace(in: app)
    }

    private func focusEditableElement(_ element: XCUIElement, in app: XCUIApplication) {
        dismissSystemBannersIfNeeded()
        app.activate()
        let focusTarget: XCUIElement
        if element.identifier == "workspace-search-editor" || element.identifier == "workspace-search-text-field" {
            let searchLabel = app.staticTexts["workspace-search-text-field"]
            focusTarget = searchLabel.exists ? searchLabel : element
        } else {
            focusTarget = element
        }

        focusTarget.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).click()
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
    }

    private func submitSearch(using element: XCUIElement, in app: XCUIApplication) {
        focusEditableElement(element, in: app)
        app.typeText("\r")
    }

    private func enterText(_ text: String, into element: XCUIElement, in app: XCUIApplication) {
        focusEditableElement(element, in: app)
        app.typeText(text)
    }

    private func searchInput(in app: XCUIApplication) -> XCUIElement {
        let editor = identifiedElement(in: app, identifier: "workspace-search-editor")
        let editorTextView = editor.descendants(matching: .textView)["workspace-search-text-field"]
        if editorTextView.exists {
            return editorTextView
        }

        let editorSearchField = editor.descendants(matching: .searchField)["workspace-search-text-field"]
        if editorSearchField.exists {
            return editorSearchField
        }

        let nestedEditorTextField = editor.descendants(matching: .textField)["workspace-search-text-field"]
        if nestedEditorTextField.exists {
            return nestedEditorTextField
        }

        let textView = app.textViews["workspace-search-text-field"]
        if textView.exists {
            return textView
        }

        let searchField = app.searchFields["workspace-search-text-field"]
        if searchField.exists {
            return searchField
        }

        let textField = app.textFields["workspace-search-text-field"]
        if textField.exists {
            return textField
        }

        let editorField = app.textFields["workspace-search-editor"]
        if editorField.exists {
            return editorField
        }

        return identifiedElement(in: app, identifier: "workspace-search-editor")
    }

    private nonisolated func terminateInterferingSystemApps() {
        for bundleIdentifier in [
            "com.semanticgallery.app",
            "com.apple.systempreferences",
            "com.apple.systemsettings",
            "cn.better365.iShotPro",
        ] {
            for application in NSRunningApplication.runningApplications(withBundleIdentifier: bundleIdentifier) {
                if application.forceTerminate() == false {
                    application.terminate()
                }
            }
        }
    }

    private nonisolated func waitForRunningApplicationsToExit(bundleIdentifier: String, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            let activeApplications = NSRunningApplication.runningApplications(withBundleIdentifier: bundleIdentifier)
                .filter { $0.isTerminated == false }
            if activeApplications.isEmpty {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.2))
        } while Date() < deadline

        return false
    }

    private func activeSemanticGalleryBundleURL() -> URL? {
        NSRunningApplication
            .runningApplications(withBundleIdentifier: "com.semanticgallery.app")
            .first(where: { $0.isTerminated == false })?
            .bundleURL
    }

    private func waitForSemanticGalleryToLaunch(timeout: TimeInterval) -> Bool {
        waitForCondition(timeout: timeout) {
            NSRunningApplication
                .runningApplications(withBundleIdentifier: "com.semanticgallery.app")
                .contains { $0.isTerminated == false }
        }
    }

    private func waitForApplicationSurface(in app: XCUIApplication, timeout: TimeInterval) -> Bool {
        waitForCondition(timeout: timeout) {
            app.windows.firstMatch.exists
                || app.buttons["open-settings-button"].exists
                || app.buttons["choose-folder-button"].exists
                || app.buttons["start-installation-button"].exists
        }
    }

    private func waitForWindowCount(in app: XCUIApplication, expected: Int, timeout: TimeInterval) -> Bool {
        waitForCondition(timeout: timeout) {
            app.windows.count == expected
        }
    }

    private func activateSemanticGalleryIfNeeded() {
        NSRunningApplication
            .runningApplications(withBundleIdentifier: "com.semanticgallery.app")
            .first(where: { $0.isTerminated == false })?
            .activate(options: [.activateAllWindows])
        var scriptError: NSDictionary?
        _ = NSAppleScript(source: "tell application id \"com.semanticgallery.app\" to activate")?
            .executeAndReturnError(&scriptError)
        RunLoop.current.run(until: Date().addingTimeInterval(0.5))
    }

    private func requestWorkspaceWindow(in app: XCUIApplication) {
        activateSemanticGalleryIfNeeded()
        RunLoop.current.run(until: Date().addingTimeInterval(0.3))

        let fileMenuItem = app.menuBars.menuBarItems["File"]
        if fileMenuItem.waitForExistence(timeout: 2) {
            fileMenuItem.click()
            let newWindowItem = app.menuItems["New Window"]
            if newWindowItem.waitForExistence(timeout: 2) {
                newWindowItem.click()
                RunLoop.current.run(until: Date().addingTimeInterval(0.6))
                return
            }
        }

        var scriptError: NSDictionary?
        let script = """
        tell application id "com.semanticgallery.app" to activate
        tell application "System Events"
            tell process "SemanticGallery"
                keystroke "n" using command down
            end tell
        end tell
        """
        _ = NSAppleScript(source: script)?.executeAndReturnError(&scriptError)
        RunLoop.current.run(until: Date().addingTimeInterval(0.6))
    }

    private func relaunchIntoWorkspace(in app: XCUIApplication) {
        if app.state != .notRunning {
            app.terminate()
        }

        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 20), app.debugDescription)

        activateSemanticGalleryIfNeeded()
        if waitForApplicationSurface(in: app, timeout: 5) == false {
            requestWorkspaceWindow(in: app)
        }
        XCTAssertTrue(waitForApplicationSurface(in: app, timeout: 15), app.debugDescription)
    }

    private func clickElement(_ element: XCUIElement, in app: XCUIApplication? = nil) {
        dismissSystemBannersIfNeeded()
        app?.activate()
        let offset: CGVector
        if element.elementType == .textField {
            offset = CGVector(dx: -0.35, dy: 0.5)
        } else {
            offset = CGVector(dx: 0.5, dy: 0.5)
        }
        element.coordinate(withNormalizedOffset: offset).click()
    }

    private func dismissSystemBannersIfNeeded() {
        for application in NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.notificationcenterui") {
            if application.forceTerminate() == false {
                application.terminate()
            }
        }
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
    }

    private func displayedText(of element: XCUIElement) -> String? {
        let label = element.label.trimmingCharacters(in: .whitespacesAndNewlines)
        if label.isEmpty == false {
            return label
        }
        guard let rawValue = element.value else {
            return nil
        }

        let description = String(describing: rawValue).trimmingCharacters(in: .whitespacesAndNewlines)
        guard description.isEmpty == false, description != "nil" else {
            return nil
        }

        if description.hasPrefix("Optional("), description.hasSuffix(")") {
            let content = description.dropFirst("Optional(".count).dropLast()
            return String(content).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }

        return description
    }

    private func normalizePathLikeText(_ text: String) -> String {
        var normalized = text
        if normalized.count > 1, normalized.hasSuffix("/") {
            normalized.removeLast()
        }
        if normalized.hasPrefix("/private/tmp/") {
            normalized.removeFirst("/private".count)
        }
        return normalized
    }

}

private extension URL {
    var fileExists: Bool {
        FileManager.default.fileExists(atPath: path)
    }
}
