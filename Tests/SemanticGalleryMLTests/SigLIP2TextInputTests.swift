import Foundation
import Testing
import Tokenizers
@testable import SemanticGalleryML

@Test
func siglipTextInputIDsMatchLegacyPaddingContract() async throws {
    let modelDirectory = try liveArtifactFixtureRoot()
        .appending(path: "mlx")
        .appending(path: "siglip2-base-patch16-224-f32")
    let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
    let tokenizerConfig = try tokenizerConfig(from: modelDirectory)
    let maxLength = tokenizerConfig["max_length"] as? Int ?? 64
    let padToken = tokenizerConfig["pad_token"] as? String
    let padTokenID = padToken.flatMap { tokenizer.convertTokenToId($0) } ?? 0

    let inputIDs = SigLIP2Support.paddedTokenIDs(
        for: "a yellow tabby cat resting on the floor",
        tokenizer: tokenizer,
        maxLength: maxLength,
        padTokenID: padTokenID
    )

    #expect(inputIDs.count == maxLength)
    #expect(Array(inputIDs.prefix(10)) == [235250, 8123, 178169, 4401, 34626, 611, 573, 6784, 1, 0])
}

private func liveArtifactFixtureRoot() throws -> URL {
    let candidates = [
        ProcessInfo.processInfo.environment["SEMANTICGALLERY_UI_TEST_ARTIFACT_FIXTURE_ROOT"],
        "/tmp/semanticgallery-ui-artifacts",
        "/Users/\(NSUserName())/.semanticgallery-ui-artifacts",
    ]
        .compactMap { $0 }
        .map { URL(filePath: $0, directoryHint: .isDirectory) }

    guard let rootURL = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path(percentEncoded: false)) }) else {
        throw NSError(
            domain: "SemanticGalleryMLTests",
            code: 11,
            userInfo: [NSLocalizedDescriptionKey: "Artifact fixture root is missing."]
        )
    }
    return rootURL
}

private func tokenizerConfig(from modelDirectory: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: modelDirectory.appending(path: "tokenizer_config.json"))
    let payload = try JSONSerialization.jsonObject(with: data)
    return payload as? [String: Any] ?? [:]
}
