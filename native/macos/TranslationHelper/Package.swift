// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TranslationHelper",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "TranslationHelper", targets: ["TranslationHelper"]),
    ],
    targets: [
        .executableTarget(name: "TranslationHelper"),
    ]
)
