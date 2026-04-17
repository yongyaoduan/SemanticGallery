// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SemanticGallery",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "SemanticGalleryAppShell", targets: ["SemanticGalleryAppShell"]),
        .library(name: "SemanticGalleryCore", targets: ["SemanticGalleryCore"]),
        .library(name: "SemanticGalleryIndex", targets: ["SemanticGalleryIndex"]),
        .library(name: "SemanticGalleryML", targets: ["SemanticGalleryML"]),
        .library(name: "SemanticGalleryPersistence", targets: ["SemanticGalleryPersistence"]),
        .library(name: "SemanticGalleryInstall", targets: ["SemanticGalleryInstall"]),
        .library(name: "SemanticGallerySearch", targets: ["SemanticGallerySearch"]),
        .library(name: "SemanticGallerySettings", targets: ["SemanticGallerySettings"]),
        .library(name: "SemanticGalleryUI", targets: ["SemanticGalleryUI"]),
        .executable(name: "SemanticGallery", targets: ["SemanticGallery"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "49d73abcbe8f49f44c1e61911997e6b680484216"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "7f1f9d06c8fc789936a4cca2affe96528e99f47d"),
    ],
    targets: [
        .executableTarget(
            name: "SemanticGallery",
            dependencies: ["SemanticGalleryAppShell"],
            path: "App/SemanticGallery",
            resources: [
                .process("Assets.xcassets"),
            ]
        ),
        .target(
            name: "SemanticGalleryAppShell",
            dependencies: [
                "SemanticGalleryCore",
                "SemanticGalleryIndex",
                "SemanticGalleryInstall",
                "SemanticGalleryML",
                "SemanticGalleryPersistence",
                "SemanticGallerySearch",
                "SemanticGallerySettings",
                "SemanticGalleryUI",
            ],
            path: "Sources/SemanticGalleryAppShell"
        ),
        .target(
            name: "SemanticGalleryCore",
            path: "Sources/SemanticGalleryCore"
        ),
        .target(
            name: "SemanticGalleryIndex",
            dependencies: ["SemanticGalleryCore", "SemanticGalleryPersistence"],
            path: "Sources/SemanticGalleryIndex"
        ),
        .target(
            name: "SemanticGalleryML",
            dependencies: [
                "SemanticGalleryCore",
                "SemanticGalleryIndex",
                "SemanticGalleryPersistence",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/SemanticGalleryML"
        ),
        .target(
            name: "SemanticGalleryPersistence",
            path: "Sources/SemanticGalleryPersistence",
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .target(
            name: "SemanticGalleryInstall",
            dependencies: ["SemanticGalleryCore", "SemanticGalleryPersistence", "SemanticGalleryUI"],
            path: "Sources/SemanticGalleryInstall"
        ),
        .target(
            name: "SemanticGallerySearch",
            dependencies: ["SemanticGalleryCore", "SemanticGalleryPersistence", "SemanticGalleryUI"],
            path: "Sources/SemanticGallerySearch"
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
            name: "SemanticGalleryAppShellTests",
            dependencies: [
                "SemanticGalleryAppShell",
                "SemanticGalleryIndex",
                "SemanticGalleryInstall",
                "SemanticGalleryML",
                "SemanticGalleryPersistence",
            ],
            path: "Tests/SemanticGalleryAppShellTests"
        ),
        .testTarget(
            name: "SemanticGalleryInstallTests",
            dependencies: ["SemanticGalleryInstall", "SemanticGalleryPersistence"],
            path: "Tests/SemanticGalleryInstallTests"
        ),
        .testTarget(
            name: "SemanticGalleryPersistenceTests",
            dependencies: ["SemanticGalleryPersistence"],
            path: "Tests/SemanticGalleryPersistenceTests"
        ),
        .testTarget(
            name: "SemanticGalleryIndexTests",
            dependencies: ["SemanticGalleryIndex", "SemanticGalleryPersistence"],
            path: "Tests/SemanticGalleryIndexTests"
        ),
        .testTarget(
            name: "SemanticGallerySearchTests",
            dependencies: ["SemanticGallerySearch", "SemanticGalleryPersistence"],
            path: "Tests/SemanticGallerySearchTests"
        ),
        .testTarget(
            name: "SemanticGalleryMLTests",
            dependencies: [
                "SemanticGalleryML",
                "SemanticGalleryPersistence",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/SemanticGalleryMLTests"
        ),
        .testTarget(
            name: "SemanticGallerySettingsTests",
            dependencies: ["SemanticGalleryCore", "SemanticGallerySettings", "SemanticGalleryPersistence"],
            path: "Tests/SemanticGallerySettingsTests"
        ),
    ]
)
