import Foundation
import MLX
import MLXNN
import MLXOptimizers

extension PrivateAdaptationTrainer {
    func trainAdapter(
        privateImages: [PrivateAdaptationImage],
        publicExamples: [PublicAnchorExample],
        preparedTrainingData: PreparedTrainingData,
        logitScale: MLXArray,
        defaults: PrivateAdaptationTrainingDefaults,
        onStep: @escaping @Sendable (Int, Int, Double) async -> Void
    ) async throws -> AdaptationTrainingResult {
        let adapter = ImageProjectionAdapter(dimension: preparedTrainingData.featureDimension, rank: 16)
        eval(adapter)
        let optimizer = AdamW(learningRate: 2e-4, weightDecay: 1e-2)
        let lossAndGrad = valueAndGrad(model: adapter) { (adapter: ImageProjectionAdapter, arrays: [MLXArray]) in
            let adaptedPublicImage = adapter(arrays[0])
            let publicLoss = AdaptationTrainingMath.contrastiveLoss(
                imageEmbeddings: adaptedPublicImage,
                textEmbeddings: arrays[1],
                logitScale: logitScale
            )

            let adaptedOriginal = adapter(arrays[2])
            let adaptedViewA = adapter(arrays[3])
            let adaptedViewB = adapter(arrays[4])
            let privateLoss = AdaptationTrainingMath.pairedImageLoss(
                firstEmbeddings: adaptedViewA,
                secondEmbeddings: adaptedViewB,
                logitScale: logitScale
            )
            let distill = AdaptationTrainingMath.distillationLoss(
                studentEmbeddings: adaptedOriginal,
                teacherEmbeddings: arrays[2]
            )

            let total = publicLoss + (0.30 * privateLoss) + (0.15 * distill)
            return [total]
        }

        let epochs = try PrivateAdaptationEpochPlanner(
            epochCount: defaults.epochCount,
            publicItemsPerEpoch: defaults.publicItemsPerEpoch,
            seed: defaults.seed
        ).makeEpochs(
            privateItems: privateImages,
            publicItems: publicExamples
        )

        var lossHistory: [Double] = []
        var stepIndex = 0
        for epoch in epochs {
            for startIndex in stride(from: 0, to: epoch.publicItems.count, by: defaults.miniBatchSize) {
                let endIndex = min(startIndex + defaults.miniBatchSize, epoch.publicItems.count)
                let publicBatch = try buildPublicBatch(
                    examples: Array(epoch.publicItems[startIndex..<endIndex]),
                    preparedExamples: preparedTrainingData.publicExamples
                )
                let privateBatch = try buildPrivateBatch(
                    images: Array(epoch.privateItems[startIndex..<endIndex]),
                    preparedImagesByPath: preparedTrainingData.privateImagesByPath
                )
                let arrays = [
                    publicBatch.imageFeatures,
                    publicBatch.textFeatures,
                    privateBatch.originalFeatures,
                    privateBatch.viewAFeatures,
                    privateBatch.viewBFeatures,
                ]

                let (values, gradients) = lossAndGrad(adapter, arrays)
                let loss = values[0]
                optimizer.update(model: adapter, gradients: gradients)
                eval(loss, adapter, optimizer)
                let lossValue = Double(loss.item(Float.self))
                lossHistory.append(lossValue)
                stepIndex += 1
                await onStep(stepIndex, defaults.totalSteps, lossValue)
            }
        }

        return AdaptationTrainingResult(adapter: adapter, lossHistory: lossHistory)
    }

    func prepareTrainingData(
        session: SigLIPLoadedSession,
        privateImages: [PrivateAdaptationImage],
        publicExamples: [PublicAnchorExample],
        batchSize: Int
    ) throws -> PreparedTrainingData {
        let preparedPublicExamples = try preparePublicExamples(
            examples: publicExamples,
            session: session,
            batchSize: batchSize
        )
        let preparedPrivateImages = try preparePrivateImages(
            images: privateImages,
            imageSize: session.config.visionConfig.imageSize,
            session: session,
            batchSize: batchSize
        )
        let featureDimension = preparedPublicExamples.values.first?.imageFeatures.shape[1]
            ?? preparedPrivateImages.values.first?.originalFeatures.shape[1]
            ?? session.config.textConfig.projectionSize

        return PreparedTrainingData(
            publicExamples: preparedPublicExamples,
            privateImagesByPath: preparedPrivateImages,
            featureDimension: featureDimension
        )
    }
}
