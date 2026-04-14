// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SemanticGallery",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SemanticGalleryCore", targets: ["SemanticGalleryCore"]),
        .executable(name: "SemanticGalleryMacApp", targets: ["SemanticGalleryMacApp"]),
    ],
    targets: [
        .executableTarget(
            name: "SemanticGalleryMacApp",
            dependencies: ["SemanticGalleryCore"],
            path: "App/SemanticGalleryMacApp"
        ),
        .target(
            name: "SemanticGalleryCore",
            path: "Sources/SemanticGalleryCore"
        ),
        .testTarget(
            name: "SemanticGalleryCoreTests",
            dependencies: ["SemanticGalleryCore"],
            path: "Tests/SemanticGalleryCoreTests"
        ),
    ]
)
