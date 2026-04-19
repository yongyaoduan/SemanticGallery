import Foundation
import MLX
import MLXNN

final class ImageProjectionAdapter: Module, UnaryLayer {
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "down") var down: MLXArray
    @ParameterInfo(key: "up") var up: MLXArray

    let dimension: Int
    let rank: Int

    init(dimension: Int, rank: Int = 16, seed: UInt64 = 20_260_405) {
        self.dimension = dimension
        self.rank = rank
        self._scale.wrappedValue = MLXArray(Array(repeating: Float(1), count: dimension), [dimension])
        self._bias.wrappedValue = MLXArray(Array(repeating: Float(0), count: dimension), [dimension])
        self._down.wrappedValue = MLXArray(Self.initialDownValues(dimension: dimension, rank: rank, seed: seed), [dimension, rank])
        self._up.wrappedValue = MLXArray(Array(repeating: Float(0), count: rank * dimension), [rank, dimension])
    }

    init(arrays: [String: MLXArray]) throws {
        guard
            let scale = arrays["scale"],
            let bias = arrays["bias"],
            let down = arrays["down"],
            let up = arrays["up"]
        else {
            throw GalleryEmbeddingError.invalidAdapterArtifact
        }

        self.dimension = scale.shape[0]
        self.rank = down.shape[1]
        self._scale.wrappedValue = scale
        self._bias.wrappedValue = bias
        self._down.wrappedValue = down
        self._up.wrappedValue = up
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let working = input.asType(SigLIPPrecisionPolicy.training)
        let scale = self.scale.asType(SigLIPPrecisionPolicy.training)
        let bias = self.bias.asType(SigLIPPrecisionPolicy.training)
        let down = self.down.asType(SigLIPPrecisionPolicy.training)
        let up = self.up.asType(SigLIPPrecisionPolicy.training)
        let residual = matmul(matmul(working, down), up)
        let adapted = (working * scale) + residual + bias
        return SigLIP2Support.normalizeEmbeddings(adapted).asType(input.dtype)
    }

    func arraysForSaving() -> [String: MLXArray] {
        [
            "scale": scale,
            "bias": bias,
            "down": down,
            "up": up,
        ]
    }

    private static func initialDownValues(dimension: Int, rank: Int, seed: UInt64) -> [Float] {
        var generator = SeededAdapterGenerator(state: seed)
        return (0..<(dimension * rank)).map { _ in
            let value = Float(generator.next() % 10_000) / 10_000
            return (value - 0.5) * 0.02
        }
    }
}

private struct SeededAdapterGenerator: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = 2_862_933_555_777_941_757 &* state &+ 3_037_000_493
        return state
    }
}
