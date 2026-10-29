import SwiftUI

@main
struct OrchardMenuBarApp: App {
    @StateObject private var model = OrchardModel()

    var body: some Scene {
        MenuBarExtra {
            MenuContent(model: model)
        } label: {
            Image(systemName: model.iconName)
                .accessibilityLabel("Orchard")
        }
        .menuBarExtraStyle(.menu)
    }
}

struct MenuContent: View {
    @ObservedObject var model: OrchardModel

    var body: some View {
        if let error = model.errorLine {
            Text(error)
            Divider()
        }
        ForEach(Array(model.workerLines.enumerated()), id: \.offset) { _, line in
            Text(line)
        }
        Text(model.queueLine)
        if !model.heldLines.isEmpty {
            Divider()
            ForEach(Array(model.heldLines.enumerated()), id: \.offset) { _, line in
                Text(line)
            }
        }
        if let active = model.activeLine {
            Divider()
            Text(active)
        }
        Divider()
        Button(model.dispatchRunning ? "Stop Dispatch" : "Start Dispatch") {
            model.toggleDispatch()
        }
        Toggle("Start Dispatch Automatically", isOn: Binding(
            get: { model.autoDispatch },
            set: { model.setAutoDispatch($0) }
        ))
        Toggle("Launch at Login", isOn: Binding(
            get: { model.launchAtLogin },
            set: { model.setLaunchAtLogin($0) }
        ))
        Button("Queue Test Job") {
            model.queueTestJob()
        }
        Button("Refresh") {
            model.refresh()
        }
        Divider()
        Button("Open Orchard Folder") {
            NSWorkspace.shared.open(URL(fileURLWithPath: NSHomeDirectory() + "/.orchard"))
        }
        Button("Quit Orchard") {
            NSApp.terminate(nil)
        }
    }
}
