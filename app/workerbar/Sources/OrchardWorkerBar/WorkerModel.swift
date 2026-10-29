import AppKit
import Foundation
import ServiceManagement

struct WorkerSnapshot {
    var paused = false
    var caps: [String: Any]?
    var jobs: [(id: String, state: String, elapsed: Double?, active: Double?, note: String?)] = []
}

@MainActor
final class WorkerModel: ObservableObject {
    @Published var iconName = "leaf"
    @Published var statusLines: [String] = ["loading…"]
    @Published var paused = false
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled

    nonisolated static let orchardDir = NSHomeDirectory() + "/.orchard"
    private var timer: Timer?
    private var refreshing = false

    init() {
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 8, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    func refresh() {
        guard !refreshing else { return }
        refreshing = true
        DispatchQueue.global(qos: .utility).async {
            let snapshot = WorkerModel.collect()
            DispatchQueue.main.async {
                self.refreshing = false
                self.apply(snapshot)
            }
        }
    }

    func togglePause() {
        var obj: [String: Any] = [:]
        let url = URL(fileURLWithPath: Self.orchardDir + "/policy.json")
        if let data = try? Data(contentsOf: url),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            obj = existing
        }
        obj["paused"] = !paused
        try? FileManager.default.createDirectory(atPath: Self.orchardDir, withIntermediateDirectories: true)
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url, options: .atomic)
        }
        paused = !paused
        refresh()
    }

    func setLaunchAtLogin(_ on: Bool) {
        do {
            if on {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            launchAtLogin = on
        } catch {
            statusLines = ["login item: \(error.localizedDescription)"] + statusLines
        }
    }

    nonisolated static func collect() -> WorkerSnapshot {
        var snapshot = WorkerSnapshot()
        let fm = FileManager.default

        let policyPath = orchardDir + "/policy.json"
        if let data = try? Data(contentsOf: URL(fileURLWithPath: policyPath)),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            snapshot.paused = obj["paused"] as? Bool ?? false
        }

        let jobsDir = orchardDir + "/jobs"
        if let names = try? fm.contentsOfDirectory(atPath: jobsDir) {
            for name in names.sorted() {
                let path = jobsDir + "/" + name + "/status.json"
                guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let state = obj["state"] as? String else { continue }
                if state == "running" || state == "paused" {
                    snapshot.jobs.append((
                        id: name,
                        state: state,
                        elapsed: obj["elapsed_s"] as? Double,
                        active: obj["active_s"] as? Double,
                        note: obj["pause_reason"] as? String
                    ))
                }
            }
        }

        let agent = orchardDir + "/bin/orchard-agent.py"
        if fm.fileExists(atPath: agent) {
            snapshot.caps = runAgentStatus(agent)
        }
        return snapshot
    }

    nonisolated static func runAgentStatus(_ agent: String) -> [String: Any]? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        proc.arguments = [agent, "status"]
        let out = Pipe()
        proc.standardOutput = out
        proc.standardError = Pipe()
        do {
            try proc.run()
        } catch {
            return nil
        }
        proc.waitUntilExit()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return obj
    }

    private func apply(_ snapshot: WorkerSnapshot) {
        paused = snapshot.paused
        var lines: [String] = []

        if let caps = snapshot.caps {
            let idleText = (caps["idle_seconds"] as? Double).map { "idle \(Self.humanDuration($0))" } ?? "idle ?"
            let power = caps["power"] as? String ?? "?"
            let disk = caps["disk_free_gb"] as? Double ?? 0
            lines.append(String(format: "This Mac: %@ · %@ · %.1f GB free", idleText, power, disk))
        } else {
            lines.append("worker not set up yet")
        }

        if snapshot.jobs.isEmpty {
            lines.append("no orchard jobs running")
        } else {
            for job in snapshot.jobs {
                let short = String(job.id.prefix(24))
                if job.state == "running" {
                    let elapsed = job.elapsed.map { Self.humanDuration($0) } ?? "?"
                    let active = job.active.map { Self.humanDuration($0) } ?? "?"
                    lines.append("\(short): running (\(elapsed), active \(active))")
                } else {
                    lines.append("\(short): paused — \(job.note ?? "waiting")")
                }
            }
        }

        if paused {
            lines.append("Orchard is paused on this Mac")
        }
        statusLines = lines

        if paused || snapshot.jobs.contains(where: { $0.state == "paused" }) {
            iconName = "pause.circle"
        } else if snapshot.jobs.contains(where: { $0.state == "running" }) {
            iconName = "hourglass"
        } else {
            iconName = "leaf"
        }
    }

    static func humanDuration(_ seconds: Double) -> String {
        let total = Int(seconds)
        if total >= 3600 { return "\(total / 3600)h\(String(format: "%02d", (total % 3600) / 60))m" }
        if total >= 60 { return "\(total / 60)m" }
        return "\(total)s"
    }
}
