// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SemanticGallery",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SemanticGalleryCore", targets: ["SemanticGalleryCore"]),
        .library(name: "SemanticGalleryPersistence", targets: ["SemanticGalleryPersistence"]),
        .library(name: "SemanticGalleryInstall", targets: ["SemanticGalleryInstall"]),
        .library(name: "SemanticGallerySettings", targets: ["SemanticGallerySettings"]),
        .library(name: "SemanticGalleryUI", targets: ["SemanticGalleryUI"]),
        .executable(name: "SemanticGalleryMacApp", targets: ["SemanticGalleryMacApp"]),
    ],
    targets: [
        .executableTarget(
            name: "SemanticGalleryMacApp",
            dependencies: ["SemanticGalleryCore", "SemanticGalleryInstall", "SemanticGallerySettings", "SemanticGalleryUI"],
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
            dependencies: ["SemanticGalleryCore", "SemanticGalleryPersistence", "SemanticGalleryUI"],
            path: "Sources/SemanticGalleryInstall"
        ),
        .target(
            name: "SemanticGallerySettings",
            dependencies: ["SemanticGalleryCore", "SemanticGalleryPersistence", "SemanticGalleryUI"],
            path: "Sources/SemanticGallerySettings"
        ),
        .target(
            name: "SemanticGalleryUI",
            path: "Sources/SemanticGalleryUI"
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
        .testTarget(
            name: "SemanticGallerySettingsTests",
            dependencies: ["SemanticGalleryCore", "SemanticGallerySettings", "SemanticGalleryPersistence"],
            path: "Tests/SemanticGallerySettingsTests"
        ),
    ]
)
