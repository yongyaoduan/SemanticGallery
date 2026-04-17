import Foundation
import SemanticGalleryPersistence

public struct InstallStateStore {
    public let paths: AppPaths
    public let catalog: ArtifactCatalog

    public init(paths: AppPaths, catalog: ArtifactCatalog = .legacyCompatible) {
        self.paths = paths
        self.catalog = catalog
    }

    public func isInstallationComplete() throws -> Bool {
        let fileManager = FileManager.default
        let artifactsBaseRoot = paths.runtimeArtifactsRoot

        func artifactComplete(
            _ artifact: RemoteArtifact,
            validator: (URL, String) -> Bool
        ) -> Bool {
            let artifactRoot = artifactsBaseRoot.appending(path: artifact.relativePath)
            return artifact.requiredFiles.allSatisfy { filename in
                let url = artifactRoot.appending(path: filename)
                let path = url.path
                guard fileManager.fileExists(atPath: path) else {
                    return false
                }
                let size = (try? fileManager.attributesOfItem(atPath: path)[.size] as? NSNumber)?.int64Value ?? 0
                guard size > 0 else {
                    return false
                }
                return validator(url, filename)
            }
        }

        guard artifactComplete(catalog.baseModel, validator: validateBaseModelFile) else { return false }
        guard artifactComplete(catalog.stage1Checkpoint, validator: validateStage1File) else { return false }
        guard artifactComplete(catalog.publicAnchor, validator: validatePublicAnchorFile) else { return false }
        guard artifactsBaseRoot.standardizedFileURL == paths.artifactsRoot.standardizedFileURL else {
            return true
        }
        guard fileManager.fileExists(atPath: paths.installStateURL.path) else {
            return false
        }
        return validateJSONObject(at: paths.installStateURL)
    }

    private func validateBaseModelFile(at url: URL, filename: String) -> Bool {
        switch filename {
        case "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json":
            return validateJSONObject(at: url)
        default:
            return true
        }
    }

    private func validateStage1File(at url: URL, filename: String) -> Bool {
        switch filename {
        case "weights.safetensors":
            return validateSafetensors(at: url)
        case "summary.json":
            return validateJSONObject(at: url)
        default:
            return true
        }
    }

    private func validatePublicAnchorFile(at url: URL, filename: String) -> Bool {
        switch filename {
        case "semanticgallery-stage2-public-anchor.tar.gz":
            return validateGzipArchive(at: url)
        case "sample_info.json":
            return validateJSONObject(at: url)
        default:
            return true
        }
    }

    private func validateJSONObject(at url: URL) -> Bool {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) else {
            return false
        }
        return object is [String: Any] || object is [Any]
    }

    private func validateGzipArchive(at url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return false
        }
        defer { try? handle.close() }
        let prefix = try? handle.read(upToCount: 2)
        return prefix == Data([0x1f, 0x8b])
    }

    private func validateSafetensors(at url: URL) -> Bool {
        guard let data = try? Data(contentsOf: url), data.count >= 8 else {
            return false
        }
        let headerLength = data.prefix(8).enumerated().reduce(UInt64.zero) { partial, item in
            partial | (UInt64(item.element) << (UInt64(item.offset) * 8))
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLength)
        guard data.count >= headerEnd else {
            return false
        }
        let headerData = data.subdata(in: headerStart..<headerEnd)
        guard let object = try? JSONSerialization.jsonObject(with: headerData),
              let dictionary = object as? [String: Any] else {
            return false
        }
        return dictionary.keys.contains(where: { $0 != "__metadata__" })
    }
}
