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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()

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
        XCTAssertTrue(app.staticTexts["Choose a folder to begin"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["Open Settings at the far right, then choose your folder"].waitForExistence(timeout: 5))
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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()

        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["choose-folder-button"].exists)
        XCTAssertTrue(identifiedElement(in: app, identifier: "workspace-path-bar").waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.staticTexts["Idle"].waitForExistence(timeout: 5))
        XCTAssertTrue(identifiedElement(in: app, identifier: "settings-library-empty-prompt").waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["Choose a library in Settings to open the archive."].exists)
        saveScreenshot(named: "workspace-empty-direct", in: app, evidenceRoot: evidenceRoot)
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

        app.launch()

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

        app.launch()

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
        searchInput.typeText("sample-01")
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

        app.launch()

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

        app.launch()
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 5))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: supportedImageCount(in: selectedFolder), timeout: 20))

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
        let previewNextButton = app.buttons["workspace-preview-next-button"]
        let previewPreviousButton = app.buttons["workspace-preview-previous-button"]
        XCTAssertTrue(previewSimilarButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewInfoButton.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertTrue(previewDeleteButton.waitForExistence(timeout: 5), app.debugDescription)
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

        app.typeKey(XCUIKeyboardKey.escape.rawValue, modifierFlags: [])
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
        XCTAssertTrue(app.staticTexts["\(supportedImageCount(in: selectedFolder)) selected"].waitForExistence(timeout: 5))
        clickElement(selectionToggleButton, in: app)
        XCTAssertTrue(app.staticTexts["0 selected"].waitForExistence(timeout: 5))
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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()

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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()

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

        XCTAssertTrue(installedSemanticGalleryAppURL().fileExists)
        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))

        let app = attachedInstalledApp()
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 20), app.debugDescription)
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            installedSemanticGalleryAppURL().standardizedFileURL
        )

        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 20), app.debugDescription)
        assertBundledArtifactsAvailable(at: installedSemanticGalleryAppURL())
        saveScreenshot(named: "01-installed-workspace-empty", in: app, evidenceRoot: evidenceRoot)

        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 10))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        XCTAssertTrue(waitForSettingsBusyState(in: app, title: "Indexing", timeout: 10), app.debugDescription)
        saveScreenshot(named: "02a-installed-after-folder-choice", in: app, evidenceRoot: evidenceRoot)
        XCTAssertTrue(waitForSettingsReadyState(in: app, timeout: 120), app.debugDescription)
        closeSettingsWindowIfPresent(in: app)
        XCTAssertTrue(waitForResultCellCount(in: app, expected: supportedImageCount(in: selectedFolder), timeout: 30))
        saveScreenshot(named: "03-installed-workspace", in: app, evidenceRoot: evidenceRoot)
    }

    func testInstalledDMGAppIndexesAndSearchesARealAlbumFolderWithoutTemporaryRuntimeRoots() throws {
        let selectedFolder = try realWorldAlbumFolder()
        let supportedImages = supportedImageURLs(in: selectedFolder)
        guard supportedImages.count >= 2 else {
            XCTFail("The real-world album folder needs at least two supported images.")
            return
        }

        let textQueryImage = supportedImages[0]
        let imageQueryImage = supportedImages[1]
        let expectedImageCount = supportedImages.count

        try removeInstalledAppArtifacts()
        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))

        let app = attachedInstalledApp()
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 20), app.debugDescription)
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            installedSemanticGalleryAppURL().standardizedFileURL
        )

        assertBundledArtifactsAvailable(at: installedSemanticGalleryAppURL())
        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 20), app.debugDescription)
        clickElement(app.buttons["open-settings-button"], in: app)
        XCTAssertTrue(app.buttons["settings-choose-folder-button"].waitForExistence(timeout: 10))
        clickElement(app.buttons["settings-choose-folder-button"], in: app)
        chooseFolderThroughOpenPanel(in: app, folderURL: selectedFolder)

        let databaseURL = actualAppSupportRoot().appending(path: "library.sqlite")
        let logURL = actualLogsRoot().appending(path: "semanticgallery.log")

        XCTAssertTrue(
            waitForCondition(timeout: 180) {
                (try? self.scalarCount(in: databaseURL, sql: "SELECT COUNT(*) FROM file_instances")) == expectedImageCount
            },
            "The selected folder was not fully indexed in the installed app database."
        )
        XCTAssertTrue(
            waitForCondition(timeout: 180) {
                (try? self.scalarCount(in: databaseURL, sql: "SELECT COUNT(*) FROM embeddings")) == expectedImageCount
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
            try scalarCount(
                in: databaseURL,
                sql: "SELECT COUNT(*) FROM embeddings"
            ),
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
        searchInput.typeText(textQueryImage.deletingPathExtension().lastPathComponent)
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

        _ = try mountedDMGAppURL()
        try installSemanticGalleryAppFromMountedDMG()
        terminateInterferingSystemApps()
        XCTAssertTrue(waitForRunningApplicationsToExit(bundleIdentifier: "com.semanticgallery.app", timeout: 10))

        let app = attachedInstalledApp()
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 20), app.debugDescription)
        XCTAssertEqual(
            activeSemanticGalleryBundleURL()?.standardizedFileURL,
            installedSemanticGalleryAppURL().standardizedFileURL
        )

        XCTAssertFalse(app.buttons["start-installation-button"].exists)
        XCTAssertTrue(app.buttons["open-settings-button"].waitForExistence(timeout: 20), app.debugDescription)
        XCTAssertTrue(app.staticTexts["Choose a folder to begin"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Open Settings at the far right, then choose your folder"].waitForExistence(timeout: 10))
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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()
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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()
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
        searchInput.typeText("private-001")
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
            folderPreparationDelayMilliseconds: 0
        )

        app.launch()
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
        searchInput.typeText(textQuery)
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
        ]
        if deferSelfUninstallTermination {
            app.launchEnvironment["SEMANTICGALLERY_DEFER_SELF_UNINSTALL_TERMINATION"] = "1"
        }
        if let artifactSourceRoot {
            app.launchEnvironment["SEMANTICGALLERY_ARTIFACT_SOURCE_ROOT"] = artifactSourceRoot
        }
        return app
    }

    private func configuredInstalledApp(runtimeRoot: URL, suiteName: String) -> XCUIApplication {
        let app = XCUIApplication(url: installedSemanticGalleryAppURL())
        app.launchArguments.append("--semanticgallery-runtime-overrides")
        app.launchEnvironment = [
            "SEMANTICGALLERY_ENABLE_RUNTIME_OVERRIDES": "1",
            "SEMANTICGALLERY_APP_ROOT": runtimeRoot.path,
            "SEMANTICGALLERY_DEFAULTS_SUITE": suiteName,
            "SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT": fixtureRoot().path,
        ]
        return app
    }

    private func attachedInstalledApp() -> XCUIApplication {
        let app = XCUIApplication(url: installedSemanticGalleryAppURL())
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

        if let folderItem = waitForFolderItem(named: folderURL.lastPathComponent, in: openPanel.panel, timeout: 2) {
            clickElement(folderItem)
            RunLoop.current.run(until: Date().addingTimeInterval(0.2))
        } else if navigateOpenPanel(to: folderURL, in: openPanel, app: app) == false {
            XCTFail("The requested folder did not appear in the chooser.\n\(openPanel.panel.debugDescription)")
            return
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
        let settingsWindow = app.windows.matching(
            NSPredicate(format: "identifier == %@ OR title CONTAINS %@", "com_apple_SwiftUI_Settings_window", "Settings")
        ).firstMatch
        guard settingsWindow.exists else {
            return
        }

        app.activate()

        let closeButton = settingsWindow.buttons["_XCUI:CloseWindow"]
        if closeButton.waitForExistence(timeout: 2) {
            clickElement(closeButton, in: app)
        }
        if settingsWindow.exists {
            settingsWindow.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1)).click()
            app.typeKey("w", modifierFlags: .command)
        }
        _ = waitForNonExistence(of: settingsWindow, timeout: 5)
    }

    private func focusEditableElement(_ element: XCUIElement, in app: XCUIApplication) {
        dismissSystemBannersIfNeeded()
        app.activate()
        element.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).click()
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
    }

    private func submitSearch(using element: XCUIElement, in app: XCUIApplication) {
        focusEditableElement(element, in: app)
        element.typeText("\r")
    }

    private func searchInput(in app: XCUIApplication) -> XCUIElement {
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

        let editorTextField = app.textFields["workspace-search-editor"]
        if editorTextField.exists {
            return editorTextField
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

    private func activateSemanticGalleryIfNeeded() {
        NSRunningApplication
            .runningApplications(withBundleIdentifier: "com.semanticgallery.app")
            .first(where: { $0.isTerminated == false })?
            .activate(options: [.activateIgnoringOtherApps])
        RunLoop.current.run(until: Date().addingTimeInterval(0.5))
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

    private struct OpenPanelContext {
        let ownerApp: XCUIApplication
        let panel: XCUIElement
    }

    private struct SemanticSearchFixture {
        let folderURL: URL
        let expectedTopResults: [String]
    }

    private func identifiedElement(in app: XCUIApplication, identifier: String) -> XCUIElement {
        app.descendants(matching: .any)[identifier]
    }

    private func makeTemporaryDirectory() -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func fixtureFolder(named name: String) throws -> URL {
        let url = fixtureRoot().appending(path: "SemanticGalleryUITest\(name)")
        guard url.fileExists else {
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "UI test fixture folder does not exist: \(url.path)"]
            )
        }
        return url
    }

    private func semanticYellowCatFixture() throws -> SemanticSearchFixture {
        let folderURL = fixtureRoot().appending(path: "SemanticGalleryUITestSemanticYellowCat")
        let expectedTopResults = ["study-01.jpg", "study-02.jpg", "study-03.jpg"]
        let expectedFiles: Set<String> = [
            "study-01.jpg",
            "study-02.jpg",
            "study-03.jpg",
            "study-04.jpg",
            "study-05.jpg",
            "study-06.jpg",
            "study-07.jpg",
        ]
        let existingFiles = try? FileManager.default.contentsOfDirectory(atPath: folderURL.path)
        guard Set(existingFiles ?? []) == expectedFiles else {
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 9,
                userInfo: [NSLocalizedDescriptionKey: "The semantic search fixture is missing from \(folderURL.path)."]
            )
        }
        return SemanticSearchFixture(folderURL: folderURL, expectedTopResults: expectedTopResults)
    }

    private func fixtureRoot() -> URL {
        ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_FIXTURE_ROOT"]
            .map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? URL(filePath: "/tmp/semanticgallery-ui-fixtures", directoryHint: .isDirectory)
    }

    private func installedArtifactFixtureRoot() throws -> URL {
        let candidates = [
            ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT"],
            "/tmp/semanticgallery-ui-artifacts",
            "/Users/\(NSUserName())/.semanticgallery-ui-artifacts",
        ]
            .compactMap { $0?.isEmpty == false ? $0 : nil }
            .map { URL(filePath: $0, directoryHint: .isDirectory) }

        guard let url = candidates.first(where: \.fileExists) else {
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "UI test artifact fixture root does not exist."]
            )
        }
        return url
    }

    private func copyTree(from sourceRoot: URL, to destinationRoot: URL) throws {
        try FileManager.default.copyItem(at: sourceRoot, to: destinationRoot)
    }

    private func embeddingCount(in runtimeRoot: URL) throws -> Int {
        try scalarCount(
            in: appSupportRoot(in: runtimeRoot).appending(path: "library.sqlite"),
            sql: "SELECT COUNT(*) FROM embeddings"
        )
    }

    private func totalVisibleFileCount(in runtimeRoot: URL) throws -> Int {
        try scalarCount(
            in: appSupportRoot(in: runtimeRoot).appending(path: "library.sqlite"),
            sql: "SELECT COUNT(*) FROM file_instances WHERE is_present = 1"
        )
    }

    private func totalFileInstanceCount(in runtimeRoot: URL) throws -> Int {
        try scalarCount(
            in: appSupportRoot(in: runtimeRoot).appending(path: "library.sqlite"),
            sql: "SELECT COUNT(*) FROM file_instances"
        )
    }

    private func scalarCount(in databaseURL: URL, sql: String, textBindings: [String] = []) throws -> Int {
        var handle: OpaquePointer?
        guard sqlite3_open(databaseURL.path, &handle) == SQLITE_OK, let handle else {
            sqlite3_close(handle)
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Could not open the UI test database."]
            )
        }
        defer { sqlite3_close(handle) }

        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK,
              let statement else {
            sqlite3_finalize(statement)
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Could not prepare the SQL query."]
            )
        }
        defer { sqlite3_finalize(statement) }

        for (index, value) in textBindings.enumerated() {
            let bindingResult = value.withCString { pointer in
                sqlite3_bind_text(statement, Int32(index + 1), pointer, -1, SQLITE_TRANSIENT)
            }
            guard bindingResult == SQLITE_OK else {
                throw NSError(
                    domain: "SemanticGalleryUITests",
                    code: 8,
                    userInfo: [NSLocalizedDescriptionKey: "Could not bind SQL parameters."]
                )
            }
        }

        guard sqlite3_step(statement) == SQLITE_ROW else {
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "Could not read embedding row count."]
            )
        }
        return Int(sqlite3_column_int(statement, 0))
    }

    private func supportedImageCount(in folderURL: URL) -> Int {
        supportedImageURLs(in: folderURL).count
    }

    private func supportedImageURLs(in folderURL: URL) -> [URL] {
        let allowedExtensions = Set(["jpg", "jpeg", "png", "heic", "heif"])
        guard let enumerator = FileManager.default.enumerator(at: folderURL, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return []
        }
        return enumerator.compactMap { item in
            guard let fileURL = item as? URL,
                  allowedExtensions.contains(fileURL.pathExtension.lowercased()) else {
                return nil
            }
            return fileURL
        }
        .sorted { $0.path < $1.path }
    }

    private func appSupportRoot(in runtimeRoot: URL) -> URL {
        runtimeRoot.appending(path: "SemanticGallery")
    }

    private func actualAppSupportRoot() -> URL {
        userHomeRoot().appending(path: "Library/Application Support/SemanticGallery")
    }

    private func cachesRoot(in runtimeRoot: URL) -> URL {
        runtimeRoot.appending(path: "Caches/com.semanticgallery.app")
    }

    private func actualCachesRoot() -> URL {
        actualAppSupportRoot().appending(path: "Caches")
    }

    private func logsRoot(in runtimeRoot: URL) -> URL {
        runtimeRoot.appending(path: "Logs/SemanticGallery")
    }

    private func actualLogsRoot() -> URL {
        actualAppSupportRoot().appending(path: "Logs")
    }

    private func realWorldAlbumFolder() throws -> URL {
        let configuredPath = ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_REAL_FOLDER"]
        let fallbackPath = userHomeRoot().appending(path: "Pictures/SemanticGalleryXCUIAlbum").path(percentEncoded: false)
        let folderURL = URL(
            filePath: configuredPath?.isEmpty == false ? configuredPath! : fallbackPath,
            directoryHint: .isDirectory
        )
        guard folderURL.fileExists else {
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 12,
                userInfo: [NSLocalizedDescriptionKey: "The configured real album folder does not exist."]
            )
        }
        return folderURL
    }

    private func installStateFile(in runtimeRoot: URL) -> URL {
        appSupportRoot(in: runtimeRoot).appending(path: "install-state.json")
    }

    private func actualInstallStateFile() -> URL {
        actualAppSupportRoot().appending(path: "install-state.json")
    }

    private func actualPreferencesFile() -> URL {
        userHomeRoot().appending(path: "Library/Preferences/com.semanticgallery.app.plist")
    }

    private func actualSavedStateRoot() -> URL {
        userHomeRoot().appending(path: "Library/Saved Application State/com.semanticgallery.app.savedState")
    }

    private func actualContainersRoot() -> URL {
        userHomeRoot().appending(path: "Library/Containers/com.semanticgallery.app")
    }

    private func actualApplicationScriptsRoot() -> URL {
        userHomeRoot().appending(path: "Library/Application Scripts/com.semanticgallery.app")
    }

    private func actualByHostPreferenceFiles() -> [URL] {
        let byHostRoot = userHomeRoot().appending(path: "Library/Preferences/ByHost")
        guard byHostRoot.fileExists,
              let children = try? FileManager.default.contentsOfDirectory(
                at: byHostRoot,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
              ) else {
            return []
        }
        return children.filter { $0.lastPathComponent.hasPrefix("com.semanticgallery.app.") }
    }

    private func userHomeRoot() -> URL {
        let directPath = "/Users/\(NSUserName())"
        let directURL = URL(filePath: directPath, directoryHint: .isDirectory)
        if directURL.fileExists {
            return directURL
        }
        return URL(filePath: NSHomeDirectory(), directoryHint: .isDirectory)
    }

    private func installedSemanticGalleryAppURL() -> URL {
        URL(filePath: "/Applications/SemanticGallery.app", directoryHint: .isDirectory)
    }

    private func assertBundledArtifactsAvailable(
        at appURL: URL,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let bundledArtifactsRoot = appURL
            .appending(path: "Contents")
            .appending(path: "Resources")
            .appending(path: "SemanticGalleryArtifacts")
        XCTAssertTrue(
            bundledArtifactsRoot.fileExists,
            "Bundled semantic search artifacts are missing from \(appURL.path).",
            file: file,
            line: line
        )
        XCTAssertTrue(
            bundledArtifactsRoot.appending(path: "mlx/siglip2-base-patch16-224-f32/config.json").fileExists,
            "The bundled app is missing the base model config.",
            file: file,
            line: line
        )
        XCTAssertTrue(
            bundledArtifactsRoot.appending(path: "semanticgallery/stage1/weights.safetensors").fileExists,
            "The bundled app is missing the published stage1 weights.",
            file: file,
            line: line
        )
        XCTAssertTrue(
            bundledArtifactsRoot.appending(path: "semanticgallery/stage2_public_anchor/sample_info.json").fileExists,
            "The bundled app is missing the public anchor metadata.",
            file: file,
            line: line
        )
    }

    private func recentLogLines(at url: URL, count: Int = 20) -> String {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return "The runtime log file is not available at \(url.path)."
        }
        return text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .suffix(count)
            .joined(separator: "\n")
    }

    private func mountedDMGAppURL() throws -> URL {
        let volumeRoot = URL(filePath: "/Volumes", directoryHint: .isDirectory)
        let candidates = (try FileManager.default.contentsOfDirectory(
            at: volumeRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ))
            .filter { $0.lastPathComponent.hasPrefix("SemanticGallery") }
            .map { $0.appending(path: "SemanticGallery.app") }
            .filter(\.fileExists)
            .sorted {
                let leftDate = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rightDate = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                if leftDate == rightDate {
                    return $0.path < $1.path
                }
                return leftDate > rightDate
            }

        guard let url = candidates.first else {
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 11,
                userInfo: [NSLocalizedDescriptionKey: "No mounted SemanticGallery dmg volume is available."]
            )
        }
        return url
    }

    private func installSemanticGalleryAppFromMountedDMG() throws {
        let destinationURL = installedSemanticGalleryAppURL()
        if destinationURL.fileExists {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.copyItem(at: mountedDMGAppURL(), to: destinationURL)
    }

    private func removeInstalledAppArtifacts() throws {
        for url in [
            actualAppSupportRoot(),
            actualCachesRoot(),
            actualLogsRoot(),
            userHomeRoot().appending(path: "Library/Preferences/com.semanticgallery.app.plist"),
        ] where url.fileExists {
            try removeItemIfPossible(at: url)
        }

        UserDefaults.standard.removePersistentDomain(forName: "com.semanticgallery.app")

        for url in [
            userHomeRoot().appending(path: "Library/Saved Application State/com.semanticgallery.app.savedState"),
            userHomeRoot().appending(path: "Library/Containers/com.semanticgallery.app"),
            userHomeRoot().appending(path: "Library/Application Scripts/com.semanticgallery.app"),
        ] where url.fileExists {
            try removeItemIfPossible(at: url)
        }

        for url in actualByHostPreferenceFiles() {
            try removeItemIfPossible(at: url)
        }

        let runnerDataRoot = userHomeRoot()
            .appending(path: "Library/Containers/com.semanticgallery.app.uitests.xctrunner/Data")

        let runnerLibraryRoot = runnerDataRoot.appending(path: "Library")
        for url in [
            runnerLibraryRoot.appending(path: "Application Support/SemanticGallery"),
            runnerLibraryRoot.appending(path: "Caches/com.semanticgallery.app"),
            runnerLibraryRoot.appending(path: "Logs/SemanticGallery"),
            runnerLibraryRoot.appending(path: "Preferences/com.semanticgallery.app.plist"),
        ] where url.fileExists {
            try removeItemIfPossible(at: url)
        }

        let runnerTmpRoot = runnerDataRoot.appending(path: "tmp")
        if runnerTmpRoot.fileExists {
            let children = try FileManager.default.contentsOfDirectory(
                at: runnerTmpRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            for child in children {
                let ownedRoots = [
                    child.appending(path: "SemanticGallery"),
                    child.appending(path: "Caches/com.semanticgallery.app"),
                    child.appending(path: "Logs/SemanticGallery"),
                ]
                for url in ownedRoots where url.fileExists {
                    try removeItemIfPossible(at: url)
                }
            }
        }

        let userTempRoot = FileManager.default.temporaryDirectory
        if userTempRoot.fileExists {
            let children = try FileManager.default.contentsOfDirectory(
                at: userTempRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            for child in children {
                let ownedRoots = [
                    child.appending(path: "SemanticGallery"),
                    child.appending(path: "Caches/com.semanticgallery.app"),
                    child.appending(path: "Logs/SemanticGallery"),
                ]
                for url in ownedRoots where url.fileExists {
                    try removeItemIfPossible(at: url)
                }
            }
        }
    }

    private func removeItemIfPossible(at url: URL) throws {
        do {
            try FileManager.default.removeItem(at: url)
        } catch let error as CocoaError where error.code == .fileWriteNoPermission {
            return
        }
    }

    private func artifactFile(in runtimeRoot: URL, relativePath: String) -> URL {
        appSupportRoot(in: runtimeRoot).appending(path: "Artifacts").appending(path: relativePath)
    }

    private func fileSize(at url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    private func waitForResultCellCount(in app: XCUIApplication, expected: Int, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if app.buttons.matching(identifier: "workspace-result-cell").count == expected {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return false
    }

    private func waitForCondition(timeout: TimeInterval, predicate: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if predicate() {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        } while Date() < deadline

        return false
    }

    private func waitForHierarchyMarkers(
        in app: XCUIApplication,
        markers: [String],
        timeout: TimeInterval
    ) -> Bool {
        waitForCondition(timeout: timeout) {
            let hierarchy = app.debugDescription
            return markers.allSatisfy(hierarchy.contains)
        }
    }

    private func firstVisibleResultLabel(in app: XCUIApplication) -> String? {
        let cells = app.buttons.matching(identifier: "workspace-result-cell")
        guard cells.count > 0 else {
            return nil
        }
        return cells.element(boundBy: 0).label
    }

    private func visibleResultLabels(in app: XCUIApplication) -> [String] {
        let cells = app.buttons.matching(identifier: "workspace-result-cell")
        return (0..<cells.count).map { cells.element(boundBy: $0).label }
    }

    private func hittableResultLabels(in app: XCUIApplication) -> [String] {
        let cells = app.buttons.matching(identifier: "workspace-result-cell")
        return (0..<cells.count).compactMap { index in
            let cell = cells.element(boundBy: index)
            return cell.isHittable ? cell.label : nil
        }
    }

    private func pasteImage(fileURL: URL, into element: XCUIElement, in app: XCUIApplication) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        XCTAssertTrue(pasteboard.writeObjects([fileURL as NSURL]))
        focusEditableElement(element, in: app)
        app.typeKey("v", modifierFlags: .command)
    }

    private func pasteImage(data: Data, filename: String, into element: XCUIElement, in app: XCUIApplication) {
        let temporaryURL = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString)-\(filename)")
        do {
            try data.write(to: temporaryURL)
            pasteImage(fileURL: temporaryURL, into: element, in: app)
        } catch {
            XCTFail("Could not prepare image pasteboard data.")
        }
    }

    private func pasteText(_ text: String, into element: XCUIElement, in app: XCUIApplication) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        XCTAssertTrue(pasteboard.setString(text, forType: .string))
        focusEditableElement(element, in: app)
        app.typeKey("v", modifierFlags: .command)
    }

    private func dismissTextSuggestions(in app: XCUIApplication) {
        app.activate()
        app.typeKey(XCUIKeyboardKey.escape.rawValue, modifierFlags: [])
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
    }

    private func clearText(in element: XCUIElement, in app: XCUIApplication) {
        focusEditableElement(element, in: app)
        app.typeKey("a", modifierFlags: .command)
        app.typeKey(XCUIKeyboardKey.delete.rawValue, modifierFlags: [])
    }

    private func thumbnailLoadedCount(in app: XCUIApplication) -> Int {
        let cells = app.buttons.matching(identifier: "workspace-result-cell")
        return (0..<cells.count).reduce(into: 0) { count, index in
            let cell = cells.element(boundBy: index)
            if accessibilityValue(of: cell) == "thumbnail loaded" {
                count += 1
            }
        }
    }

    private func accessibilityValue(of element: XCUIElement) -> String? {
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

    private func embeddingCount(in runtimeRoot: URL, encoderVersion: String) throws -> Int {
        try scalarCount(
            in: appSupportRoot(in: runtimeRoot).appending(path: "library.sqlite"),
            sql: "SELECT COUNT(*) FROM embeddings WHERE encoder_version = ?",
            textBindings: [encoderVersion]
        )
    }

    private func latestTrainingRun(in runtimeRoot: URL) throws -> TrainingRunSnapshot? {
        let databaseURL = appSupportRoot(in: runtimeRoot).appending(path: "library.sqlite")
        var handle: OpaquePointer?
        guard sqlite3_open(databaseURL.path, &handle) == SQLITE_OK, let handle else {
            sqlite3_close(handle)
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 9,
                userInfo: [NSLocalizedDescriptionKey: "Could not open the training database."]
            )
        }
        defer { sqlite3_close(handle) }

        var statement: OpaquePointer?
        let sql = """
        SELECT status, encoder_version
        FROM training_runs
        ORDER BY id DESC
        LIMIT 1
        """
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            sqlite3_finalize(statement)
            throw NSError(
                domain: "SemanticGalleryUITests",
                code: 10,
                userInfo: [NSLocalizedDescriptionKey: "Could not prepare the training query."]
            )
        }
        defer { sqlite3_finalize(statement) }

        guard sqlite3_step(statement) == SQLITE_ROW else {
            return nil
        }
        guard
            let statusPointer = sqlite3_column_text(statement, 0),
            let encoderPointer = sqlite3_column_text(statement, 1)
        else {
            return nil
        }
        return TrainingRunSnapshot(
            status: String(cString: statusPointer),
            encoderVersion: String(cString: encoderPointer)
        )
    }

    private func adaptedWeightFiles(in runtimeRoot: URL) -> [URL] {
        files(
            named: "weights.safetensors",
            under: appSupportRoot(in: runtimeRoot).appending(path: "Models").appending(path: "Adapted")
        )
    }

    private func adaptedSummaryFiles(in runtimeRoot: URL) -> [URL] {
        files(
            named: "summary.json",
            under: appSupportRoot(in: runtimeRoot).appending(path: "Models").appending(path: "Adapted")
        )
    }

    private func files(named name: String, under root: URL) -> [URL] {
        guard FileManager.default.fileExists(atPath: root.path) else {
            return []
        }
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        return enumerator.compactMap { item in
            guard let url = item as? URL else {
                return nil
            }
            return url.lastPathComponent == name ? url : nil
        }
    }

    private func makeEvidenceDirectory(named name: String) -> URL {
        let root = ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_EVIDENCE_ROOT"]
            .map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? URL(filePath: "/tmp/semanticgallery-validation", directoryHint: .isDirectory)
        let directory = root.appending(path: sanitizedFilename(name)).appending(path: UUID().uuidString)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func saveScreenshot(named name: String, in app: XCUIApplication, evidenceRoot: URL) {
        app.activate()
        let screenshot = XCUIScreen.main.screenshot()
        let destinationURL = evidenceRoot.appending(path: "\(sanitizedFilename(name)).png")
        try? screenshot.pngRepresentation.write(to: destinationURL, options: .atomic)
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func recordMetric(
        named name: String,
        value: TimeInterval,
        evidenceRoot: URL,
        details: [String: String] = [:]
    ) {
        let destinationURL = evidenceRoot.appending(path: "metrics.jsonl")
        var payload: [String: Any] = [
            "metric": name,
            "value": value,
            "unit": "seconds",
            "captured_at": ISO8601DateFormatter().string(from: Date()),
        ]
        if details.isEmpty == false {
            payload["details"] = details
        }
        guard
            let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
            let line = String(data: data, encoding: .utf8)?.appending("\n")
        else {
            return
        }
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            guard let handle = try? FileHandle(forWritingTo: destinationURL) else {
                return
            }
            defer { try? handle.close() }
            try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
        } else {
            try? line.write(to: destinationURL, atomically: true, encoding: .utf8)
        }

        let attachment = XCTAttachment(string: line.trimmingCharacters(in: .whitespacesAndNewlines))
        attachment.name = "\(name)-metric"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func sanitizedFilename(_ text: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = text.lowercased().unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
        let collapsed = String(scalars)
            .replacingOccurrences(of: "--", with: "-")
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return collapsed.isEmpty ? "evidence" : collapsed
    }
}

private extension URL {
    var fileExists: Bool {
        FileManager.default.fileExists(atPath: path)
    }
}

private struct TrainingRunSnapshot {
    let status: String
    let encoderVersion: String
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
