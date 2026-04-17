import Foundation
import MLX
import Testing
@testable import SemanticGalleryML

@Test
func siglipConfigurationUsesLegacyDefaultsForMinimalConfig() throws {
    let payload = Data(
        """
        {
          "text_config": {
            "vocab_size": 256000
          },
          "vision_config": {
            "model_type": "siglip_vision_model"
          }
        }
        """.utf8
    )

    let configuration = try JSONDecoder().decode(SigLIPConfiguration.self, from: payload)

    #expect(configuration.textConfig.hiddenSize == 768)
    #expect(configuration.textConfig.maxPositionEmbeddings == 64)
    #expect(configuration.visionConfig.imageSize == 224)
    #expect(configuration.visionConfig.patchSize == 16)
}

@Test
func siglipModelNormalizesDownloadedBaseModelKeys() {
    #expect(
        SigLIPModel.normalizedWeightKey("text_model.embeddings.token_embedding.weight")
            == "text_model.text_model.embeddings.token_embedding.weight"
    )
    #expect(
        SigLIPModel.normalizedWeightKey("vision_model.embeddings.position_embedding.weight")
            == "vision_model.vision_model.embeddings.position_embedding.weight"
    )
    #expect(
        SigLIPModel.normalizedWeightKey("vision_model.head.attention.in_proj.weight")
            == "vision_model.vision_model.head.attention.in_proj.weight"
    )
    #expect(
        SigLIPModel.normalizedWeightKey("logit_scale")
            == "logit_scale"
    )
}

@Test
func siglipRuntimeUsesBFloat16ForTrainingAndDeployment() {
    #expect(SigLIPPrecisionPolicy.deployment == .bfloat16)
    #expect(SigLIPPrecisionPolicy.training == .bfloat16)
}
