// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "OrchardMenuBar",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "OrchardMenuBar",
            path: "Sources/OrchardMenuBar"
        )
    ]
)
