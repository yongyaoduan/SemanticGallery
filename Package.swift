// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SemanticGallery",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SemanticGalleryCore", targets: ["SemanticGalleryCore"]),
        .library(name: "SemanticGalleryPersistence", targets: ["SemanticGalleryPersistence"]),
        .library(name: "SemanticGalleryInstall", targets: ["SemanticGalleryInstall"]),
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
        .target(
            name: "SemanticGalleryPersistence",
            path: "Sources/SemanticGalleryPersistence"
        ),
        .target(
            name: "SemanticGalleryInstall",
            dependencies: ["SemanticGalleryPersistence"],
            path: "Sources/SemanticGalleryInstall"
        ),
        .testTarget(
            name: "SemanticGalleryCoreTests",
            dependencies: ["SemanticGalleryCore"],
            path: "Tests/SemanticGalleryCoreTests"
        ),
        .testTarget(
            name: "SemanticGalleryInstallTests",
            dependencies: ["SemanticGalleryInstall", "SemanticGalleryPersistence"],
            path: "Tests/SemanticGalleryInstallTests"
        ),
    ]
)
