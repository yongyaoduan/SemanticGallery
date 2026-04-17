import Foundation
import Testing
@testable import SemanticGalleryInstall
@testable import SemanticGalleryPersistence

private actor ArtifactTransferRecorder {
    private var items: [ArtifactTransferProgress] = []

    func append(_ item: ArtifactTransferProgress) {
        items.append(item)
    }

    func values() -> [ArtifactTransferProgress] {
        items
    }
}

@Test
func artifactDownloaderBuildsLegacyCompatibleResolveURLs() throws {
    let downloader = ArtifactDownloader()

    let baseModel = RemoteArtifact(
        repositoryID: "google/siglip2-base-patch16-224",
        repositoryType: .model,
        relativePath: "mlx/siglip2-base-patch16-224-f32",
        requiredFiles: ["config.json"]
    )
    let publicAnchor = RemoteArtifact(
        repositoryID: "Lucas20250626/semanticgallery-stage2-public-anchor",
        repositoryType: .dataset,
        relativePath: "semanticgallery/stage2_public_anchor",
        requiredFiles: ["sample_info.json"]
    )

    #expect(
        try downloader.sourceURL(for: baseModel, filename: "config.json").absoluteString
            == "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/config.json"
    )
    #expect(
        try downloader.sourceURL(for: publicAnchor, filename: "sample_info.json").absoluteString
            == "https://huggingface.co/datasets/Lucas20250626/semanticgallery-stage2-public-anchor/resolve/main/sample_info.json"
    )
}

@Test
func artifactDownloaderFetchesPublishedMetadataFromHuggingFace() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let downloader = ArtifactDownloader()

    try await downloader.download(
        artifact: RemoteArtifact(
            repositoryID: "google/siglip2-base-patch16-224",
            repositoryType: .model,
            relativePath: "mlx/siglip2-base-patch16-224-f32",
            requiredFiles: ["config.json"]
        ),
        into: root
    )
    try await downloader.download(
        artifact: RemoteArtifact(
            repositoryID: "Lucas20250626/semanticgallery-mlx-siglip2-stage1",
            repositoryType: .model,
            relativePath: "semanticgallery/stage1",
            requiredFiles: ["summary.json"]
        ),
        into: root
    )
    try await downloader.download(
        artifact: RemoteArtifact(
            repositoryID: "Lucas20250626/semanticgallery-stage2-public-anchor",
            repositoryType: .dataset,
            relativePath: "semanticgallery/stage2_public_anchor",
            requiredFiles: ["sample_info.json"]
        ),
        into: root
    )

    #expect(
        root
            .appending(path: "mlx")
            .appending(path: "siglip2-base-patch16-224-f32")
            .appending(path: "config.json")
            .fileExists
    )
    #expect(
        root
            .appending(path: "semanticgallery")
            .appending(path: "stage1")
            .appending(path: "summary.json")
            .fileExists
    )
    #expect(
        root
            .appending(path: "semanticgallery")
            .appending(path: "stage2_public_anchor")
            .appending(path: "sample_info.json")
            .fileExists
    )
}

@Test
func installCoordinatorCopiesConfiguredArtifactSourcesIntoRuntime() async throws {
    let fixtureRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let runtimeRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: fixtureRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: runtimeRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: fixtureRoot) }
    defer { try? FileManager.default.removeItem(at: runtimeRoot) }

    let catalog = ArtifactCatalog.legacyCompatible
    try writeFixtureFiles(for: catalog.baseModel, into: fixtureRoot)
    try writeFixtureFiles(for: catalog.stage1Checkpoint, into: fixtureRoot)
    try writeFixtureFiles(for: catalog.publicAnchor, into: fixtureRoot)

    let downloader = ArtifactDownloader(
        mode: .live,
        fileSources: ArtifactFileSourceCatalog(root: fixtureRoot, catalog: catalog).fileSources
    )
    let coordinator = InstallCoordinator(paths: AppPaths(root: runtimeRoot), downloader: downloader, catalog: catalog)

    _ = try await coordinator.prepare()

    let installedRoot = AppPaths(root: runtimeRoot).artifactsRoot
    let configPayload = try #require(
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: installedRoot.appending(path: "mlx").appending(path: "siglip2-base-patch16-224-f32").appending(path: "config.json"))
        ) as? [String: Any]
    )
    #expect(configPayload["fixture"] as? String == "config.json")

    let summaryPayload = try #require(
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: installedRoot.appending(path: "semanticgallery").appending(path: "stage1").appending(path: "summary.json"))
        ) as? [String: Any]
    )
    #expect(summaryPayload["fixture"] as? String == "summary.json")

    let sampleInfoPayload = try #require(
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: installedRoot.appending(path: "semanticgallery").appending(path: "stage2_public_anchor").appending(path: "sample_info.json"))
        ) as? [String: Any]
    )
    #expect(sampleInfoPayload["fixture"] as? String == "sample_info.json")
}

@Test
func artifactDownloaderStreamsIntermediateProgressForRemoteFiles() async throws {
    let fixtureRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let runtimeRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: fixtureRoot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: runtimeRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: fixtureRoot) }
    defer { try? FileManager.default.removeItem(at: runtimeRoot) }

    let payload = Data(repeating: 0x41, count: 1_048_576)
    let payloadURL = fixtureRoot.appending(path: "payload.bin")
    try payload.write(to: payloadURL)

    let server = try LocalFixtureHTTPServer(root: fixtureRoot)
    defer { server.stop() }

    let artifact = RemoteArtifact(
        repositoryID: "local/streamed-payload",
        relativePath: "semanticgallery/test-stream",
        requiredFiles: ["payload.bin"]
    )
    let downloader = ArtifactDownloader(
        mode: .live,
        fileSources: [
            artifact.relativePath: [
                "payload.bin": try #require(URL(string: "\(server.baseURL)/payload.bin"))
            ]
        ]
    )

    let recorder = ArtifactTransferRecorder()
    try await downloader.download(artifact: artifact, into: runtimeRoot) { item in
        await recorder.append(item)
    }

    let items = await recorder.values()
    #expect(items.count >= 3)
    #expect(items.contains(where: { $0.currentFileFraction > 0 && $0.currentFileFraction < 1 }))
    #expect(items.last?.completedFileCount == 1)
    #expect(items.last?.totalFileCount == 1)
    #expect(
        runtimeRoot
            .appending(path: artifact.relativePath)
            .appending(path: "payload.bin")
            .fileExists
    )
}

private extension URL {
    var fileExists: Bool {
        FileManager.default.fileExists(atPath: path)
    }
}

private final class LocalFixtureHTTPServer {
    let process: Process
    let pipe: Pipe
    let baseURL: String

    init(root: URL) throws {
        process = Process()
        pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.standardOutput = pipe
        process.standardError = Pipe()
        process.arguments = [
            "-c",
            #"""
import http.server
import os
import socketserver
import sys
import time

root = sys.argv[1]

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/payload.bin":
            self.send_error(404)
            return
        path = os.path.join(root, "payload.bin")
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(32768)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.01)

    def log_message(self, format, *args):
        return

with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
    print(server.server_address[1], flush=True)
    server.serve_forever()
"""#,
            root.path(percentEncoded: false),
        ]

        try process.run()

        let line = try #require(
            String(data: pipe.fileHandleForReading.availableData, encoding: .utf8)?
                .split(separator: "\n")
                .first
        )
        baseURL = "http://127.0.0.1:\(line)"
    }

    func stop() {
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
    }
}

private func writeFixtureFiles(for artifact: RemoteArtifact, into root: URL) throws {
    let artifactRoot = root.appending(path: artifact.relativePath)
    try FileManager.default.createDirectory(at: artifactRoot, withIntermediateDirectories: true)
    for filename in artifact.requiredFiles {
        try fixtureContents(for: artifact, filename: filename).write(to: artifactRoot.appending(path: filename))
    }
}

private func fixtureContents(for artifact: RemoteArtifact, filename: String) -> Data {
    if filename.hasSuffix(".json") {
        let payload: Any
        switch filename {
        case "config.json":
            payload = [
                "fixture": filename,
                "text_config": ["max_position_embeddings": 64],
                "vision_config": ["image_size": 224],
            ]
        case "tokenizer.json":
            payload = [
                "fixture": filename,
                "version": "1.0",
                "model": ["type": "WordPiece"],
            ]
        case "tokenizer_config.json":
            payload = [
                "fixture": filename,
                "max_length": 64,
                "pad_token": "[PAD]",
            ]
        case "special_tokens_map.json":
            payload = [
                "fixture": filename,
                "pad_token": "[PAD]",
            ]
        case "preprocessor_config.json":
            payload = [
                "fixture": filename,
                "size": 224,
                "do_resize": true,
            ]
        case "summary.json", "sample_info.json":
            payload = ["fixture": filename]
        default:
            payload = ["fixture": filename]
        }
        return (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data("{}".utf8)
    }

    if filename == "weights.safetensors" {
        let headerObject: [String: Any] = [
            "fixture_tensor": [
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            ]
        ]
        let headerData = (try? JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])) ?? Data("{}".utf8)
        let headerLength = UInt64(headerData.count)
        let prefix = withUnsafeBytes(of: headerLength.littleEndian) { Data($0) }
        return prefix + headerData + Data([0x00, 0x00, 0x00, 0x00])
    }

    if filename.hasSuffix(".tar.gz") {
        return Data([0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    }

    return Data("fixture:\(filename)".utf8)
}
