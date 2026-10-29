import AppKit
import Darwin
import Foundation
import ServiceManagement
import UserNotifications

enum OrchardCLI {
    static func findRepo() -> URL {
        let appJSON = URL(fileURLWithPath: NSHomeDirectory() + "/.orchard/app.json")
        if let data = try? Data(contentsOf: appJSON),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let path = obj["repo"] as? String {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: NSHomeDirectory() + "/orchard")
    }

    static func run(_ args: [String], repo: URL, cwd: URL? = nil, timeout: TimeInterval = 45) -> (Int32, Data) {
        let proc = Process()
        proc.executableURL = repo.appendingPathComponent("bin/orchard")
        proc.arguments = args
        proc.currentDirectoryURL = cwd ?? URL(fileURLWithPath: NSHomeDirectory())
        // Output goes to a file, not a Pipe read after exit: once `status --json`
        // passed the 64 KB pipe buffer (~100 KB with 96 queue entries, 2026-10-24)
        // the child blocked on write, every refresh timed out, and auto-start of
        // dispatch never ran. stderr was an unread Pipe too.
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("orchard-\(UUID().uuidString).out")
        guard FileManager.default.createFile(atPath: outURL.path, contents: nil),
              let outHandle = try? FileHandle(forWritingTo: outURL) else {
            return (-1, Data("cannot create \(outURL.path)".utf8))
        }
        defer {
            try? outHandle.close()
            try? FileManager.default.removeItem(at: outURL)
        }
        proc.standardOutput = outHandle
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
        } catch {
            let message = "cannot launch orchard: \(error.localizedDescription)"
            return (-1, Data(message.utf8))
        }
        let deadline = Date().addingTimeInterval(timeout)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if proc.isRunning {
            proc.terminate()
            proc.waitUntilExit()
            return (-2, Data("timed out after \(Int(timeout))s".utf8))
        }
        let data = (try? Data(contentsOf: outURL)) ?? Data()
        return (proc.terminationStatus, data)
    }
}

struct StatusPayload {
    var workers: [[String: Any]]
    var queue: [[String: Any]]
    var jobs: [[String: Any]]
    var dispatchPid: Int?
}

@MainActor
final class OrchardModel: ObservableObject {
    @Published var iconName = "leaf"
    @Published var workerLines: [String] = []
    @Published var queueLine = "queue: …"
    @Published var heldLines: [String] = []
    @Published var activeLine: String?
    @Published var errorLine: String?
    @Published var dispatchRunning = false
    @Published var launchAtLogin = SMAppService.mainApp.status == .enabled
    @Published var autoDispatch = UserDefaults.standard.object(forKey: "autoDispatch") as? Bool ?? true

    let repo: URL
    private var timer: Timer?
    private var dispatchProcess: Process?
    private var dispatchPidValue: Int?
    private var primed = false
    private var didAutoStart = false
    private var seenTerminal: Set<String> = []
    private var notificationsReady = false

    init() {
        repo = OrchardCLI.findRepo()
        if Bundle.main.bundleIdentifier != nil {
            UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
                DispatchQueue.main.async { self.notificationsReady = granted }
            }
        }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 12, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    private var refreshing = false

    func refresh() {
        guard !refreshing else { return }
        refreshing = true
        let repo = self.repo
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let (code, data) = OrchardCLI.run(["status", "--json"], repo: repo, timeout: 45)
            DispatchQueue.main.async {
                guard let self else { return }
                self.refreshing = false
                if code == 0, let payload = Self.parseStatus(data) {
                    self.apply(payload)
                } else {
                    let detail = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
                    self.errorLine = "orchard: \(detail?.isEmpty == false ? detail! : "status failed (\(code))")"
                    self.iconName = "exclamationmark.triangle"
                }
            }
        }
    }

    func toggleDispatch() {
        if let pid = dispatchPidValue {
            kill(pid_t(pid), SIGTERM)
            dispatchPidValue = nil
            dispatchRunning = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.refresh() }
            return
        }
        startDispatch()
    }

    func startDispatch() {
        let proc = Process()
        proc.executableURL = repo.appendingPathComponent("bin/orchard")
        proc.arguments = ["dispatch", "--watch"]
        proc.currentDirectoryURL = URL(fileURLWithPath: NSHomeDirectory())
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            dispatchProcess = proc
            dispatchRunning = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self.refresh() }
        } catch {
            errorLine = "cannot start dispatch: \(error.localizedDescription)"
        }
    }

    func setLaunchAtLogin(_ on: Bool) {
        do {
            if on {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            launchAtLogin = on
            errorLine = nil
        } catch {
            errorLine = "login item: \(error.localizedDescription)"
        }
    }

    func setAutoDispatch(_ on: Bool) {
        autoDispatch = on
        UserDefaults.standard.set(on, forKey: "autoDispatch")
    }

    func queueTestJob() {
        let repo = self.repo
        let playground = URL(fileURLWithPath: NSHomeDirectory() + "/.orchard/playground", isDirectory: true)
        try? FileManager.default.createDirectory(at: playground, withIntermediateDirectories: true)
        DispatchQueue.global(qos: .utility).async { [weak self] in
            _ = OrchardCLI.run(
                ["run", "--background", "--", "/bin/sh", "-c", "sleep 8; echo app-test > out/app-test.txt"],
                repo: repo,
                cwd: playground,
                timeout: 90
            )
            DispatchQueue.main.async {
                self?.refresh()
                self?.notify(title: "Orchard", body: "Test job queued (start dispatch if stopped)")
            }
        }
    }

    private static func parseStatus(_ data: Data) -> StatusPayload? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let workers = obj["workers"] as? [[String: Any]] ?? []
        let queue = obj["queue"] as? [[String: Any]] ?? []
        let jobs = obj["jobs"] as? [[String: Any]] ?? []
        let pid = obj["dispatch_pid"] as? Int
        return StatusPayload(workers: workers, queue: queue, jobs: jobs, dispatchPid: pid)
    }

    private func apply(_ payload: StatusPayload) {
        errorLine = nil
        applyWorkers(payload.workers)
        applyQueue(payload.queue)
        applyJobs(payload.jobs)
        dispatchPidValue = payload.dispatchPid
        dispatchRunning = payload.dispatchPid != nil
        iconName = computeIcon()
        if !didAutoStart {
            didAutoStart = true
            if autoDispatch && payload.dispatchPid == nil {
                startDispatch()
            }
        }
    }

    private func applyWorkers(_ workers: [[String: Any]]) {
        var lines: [String] = []
        for w in workers {
            let name = w["name"] as? String ?? "?"
            if let err = w["error"] as? String, !err.isEmpty {
                lines.append("\(name): \(err)")
                continue
            }
            guard let caps = w["caps"] as? [String: Any] else { continue }
            let idleText: String
            if let idle = caps["idle_seconds"] as? Double {
                idleText = "idle \(Self.humanDuration(idle))"
            } else {
                idleText = "idle ?"
            }
            let power = caps["power"] as? String ?? "?"
            let disk = caps["disk_free_gb"] as? Double ?? 0
            let bench = caps["bench"] as? Double ?? 0
            var suffix = ""
            if let paused = caps["policy_paused"] as? Bool, paused { suffix = " · paused by owner" }
            lines.append(String(format: "%@: %@ · %@ · %.1f GB free · %.0fM", name, idleText, power, disk, bench / 1e6) + suffix)
        }
        workerLines = lines
    }

    private func applyQueue(_ queue: [[String: Any]]) {
        lastQueue = queue
        var counts: [String: Int] = [:]
        for entry in queue {
            let state = entry["state"] as? String ?? "?"
            counts[state, default: 0] += 1
        }
        let summary = counts.keys.sorted().map { "\(counts[$0]!) \($0)" }.joined(separator: " · ")
        queueLine = "queue: \(summary.isEmpty ? "empty" : summary)"

        var held: [String] = []
        var active: String?
        for entry in queue {
            let id = entry["id"] as? String ?? "?"
            let state = entry["state"] as? String ?? "?"
            let note = entry["note"] as? String ?? ""
            let short = String(id.prefix(28))
            if state == "queued" {
                held.append("• \(short)\(note.isEmpty ? "" : " — \(note)")")
            } else if state == "running" {
                active = "running: \(short)\(note.isEmpty ? "" : " — \(note)")"
            }
        }
        heldLines = Array(held.prefix(4))
        activeLine = active
    }

    private func applyJobs(_ jobs: [[String: Any]]) {
        let terminal: Set<String> = ["succeeded", "failed", "timed_out", "killed"]
        var newTerminal: [(String, String)] = []
        for job in jobs {
            guard let id = job["id"] as? String, let state = job["state"] as? String else { continue }
            if terminal.contains(state) && !seenTerminal.contains(id) {
                seenTerminal.insert(id)
                newTerminal.append((id, state))
            }
        }
        if primed {
            for (id, state) in newTerminal {
                let worker = jobs.first { ($0["id"] as? String) == id }?["worker"] as? String ?? "?"
                notify(title: "Orchard job \(state)", body: "\(String(id.prefix(40))) on \(worker)")
            }
        }
        primed = true
    }

    private func computeIcon() -> String {
        if errorLine != nil { return "exclamationmark.triangle" }
        let paused = lastQueue.contains { ($0["note"] as? String)?.contains("paused") == true }
        if paused { return "pause.circle" }
        if lastQueue.contains(where: { ($0["state"] as? String) == "running" }) { return "hourglass" }
        if lastQueue.contains(where: { ($0["state"] as? String) == "queued" }) { return "tray.full" }
        return "leaf"
    }

    private var lastQueue: [[String: Any]] = []

    private func notify(title: String, body: String) {
        guard notificationsReady else { return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    static func humanDuration(_ seconds: Double) -> String {
        let total = Int(seconds)
        if total >= 3600 { return "\(total / 3600)h\(String(format: "%02d", (total % 3600) / 60))m" }
        if total >= 60 { return "\(total / 60)m" }
        return "\(total)s"
    }
}
