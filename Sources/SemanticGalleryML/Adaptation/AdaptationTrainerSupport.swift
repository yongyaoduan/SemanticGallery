import Foundation
import MLX

struct PublicAnchorExample: Sendable, Equatable, Hashable {
    let imageURL: URL
    let caption: String
}

struct PublicBatch {
    let imageFeatures: MLXArray
    let textFeatures: MLXArray
}

struct PrivateBatch {
    let originalFeatures: MLXArray
    let viewAFeatures: MLXArray
    let viewBFeatures: MLXArray
}

struct PreparedPublicExample {
    let imageFeatures: MLXArray
    let textFeatures: MLXArray
}

struct PreparedPrivateImage {
    let originalFeatures: MLXArray
    let viewAFeatures: MLXArray
    let viewBFeatures: MLXArray
}

struct PreparedTrainingData {
    let publicExamples: [PublicAnchorExample: PreparedPublicExample]
    let privateImagesByPath: [String: PreparedPrivateImage]
    let featureDimension: Int
}

enum PreparedTrainingDataError: Error {
    case missingPublicExample
    case missingPrivateImage
}

struct PrivateAdaptationTrainingDefaults {
    let epochCount = 10
    let publicPoolSize = 1_000
    let publicItemsPerEpoch = 100
    let miniBatchSize = 10
    let featureBatchSize = 20
    let seed: UInt64 = 20_260_417

    var totalSteps: Int {
        epochCount * Int(ceil(Double(publicItemsPerEpoch) / Double(miniBatchSize)))
    }
}

struct AdaptationTrainingResult {
    let adapter: ImageProjectionAdapter
    let lossHistory: [Double]
}

struct AdaptationSummary: Codable {
    let encoderVersion: String
    let folderPath: String
    let folderKey: String
    let privateImageCount: Int
    let publicExampleCount: Int
    let privateBatchSize: Int
    let totalSteps: Int
    let losses: [Double]
    let precision: String
    let trainedAt: String
}

struct AdaptationSeededGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = 6_364_136_223_846_793_005 &* state &+ 1
        return state
    }
}
