import SwiftUI

@main
struct OrchardWorkerBarApp: App {
    @StateObject private var model = WorkerModel()

    var body: some Scene {
        MenuBarExtra {
            WorkerMenuContent(model: model)
        } label: {
            Image(systemName: model.iconName)
                .accessibilityLabel("Orchard Worker")
        }
        .menuBarExtraStyle(.menu)
    }
}

struct WorkerMenuContent: View {
    @ObservedObject var model: WorkerModel

    var body: some View {
        ForEach(Array(model.statusLines.enumerated()), id: \.offset) { _, line in
            Text(line)
        }
        Divider()
        Button(model.paused ? "Resume Orchard" : "Pause Orchard") {
            model.togglePause()
        }
        Toggle("Launch at Login", isOn: Binding(
            get: { model.launchAtLogin },
            set: { model.setLaunchAtLogin($0) }
        ))
        Button("Refresh") {
            model.refresh()
        }
        Divider()
        Button("Open Orchard Folder") {
            NSWorkspace.shared.open(URL(fileURLWithPath: WorkerModel.orchardDir))
        }
        Button("Quit Orchard Worker") {
            NSApp.terminate(nil)
        }
    }
}
