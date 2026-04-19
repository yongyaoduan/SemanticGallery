import AppKit
import XCTest
import SQLite3

private struct OpenPanelContext {
    let ownerApp: XCUIApplication
    let panel: XCUIElement
}

private struct SemanticSearchFixture {
    let folderURL: URL
    let expectedTopResults: [String]
}

private struct TrainingRunSnapshot {
    let status: String
    let encoderVersion: String
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

@MainActor
extension PhaseAFlowTests {
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
        appSupportRoot(in: runtimeRoot).appending(path: "Caches")
    }

    private func actualCachesRoot() -> URL {
        actualAppSupportRoot().appending(path: "Caches")
    }

    private func logsRoot(in runtimeRoot: URL) -> URL {
        appSupportRoot(in: runtimeRoot).appending(path: "Logs")
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
            _ = try? handle.seekToEnd()
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
