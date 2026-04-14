import Testing
@testable import SemanticGalleryInstall

@Test
func artifactCatalogMatchesLegacyRepositories() {
    let catalog = ArtifactCatalog.legacyCompatible
    #expect(catalog.baseModel.repositoryID == "google/siglip2-base-patch16-224")
    #expect(catalog.stage1Checkpoint.repositoryID == "Lucas20250626/semanticgallery-mlx-siglip2-stage1")
    #expect(catalog.publicAnchor.repositoryID == "Lucas20250626/semanticgallery-stage2-public-anchor")
}
