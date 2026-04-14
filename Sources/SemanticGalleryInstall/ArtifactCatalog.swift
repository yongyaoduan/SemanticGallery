public extension ArtifactCatalog {
    static let legacyCompatible = ArtifactCatalog(
        baseModel: RemoteArtifact(
            repositoryID: "google/siglip2-base-patch16-224",
            relativePath: "mlx/siglip2-base-patch16-224-f32",
            requiredFiles: [
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
            ]
        ),
        stage1Checkpoint: RemoteArtifact(
            repositoryID: "Lucas20250626/semanticgallery-mlx-siglip2-stage1",
            relativePath: "semanticgallery/stage1",
            requiredFiles: [
                "weights.safetensors",
                "summary.json",
            ]
        ),
        publicAnchor: RemoteArtifact(
            repositoryID: "Lucas20250626/semanticgallery-stage2-public-anchor",
            relativePath: "semanticgallery/stage2_public_anchor",
            requiredFiles: [
                "semanticgallery-stage2-public-anchor.tar.gz",
                "sample_info.json",
            ]
        )
    )
}
