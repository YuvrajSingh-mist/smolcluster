/// Grove P2P helper — AWDL transport sidecar.
/// Modes: discover (browse for coordinators) and mesh (full peer connections).

import Foundation
import Network

let OP_SEND: UInt8 = 0x01
let OP_RECV: UInt8 = 0x02
let OP_DISCONNECT: UInt8 = 0x03
let serviceType = "_grove._tcp"

let p2pParams: NWParameters = {
    let p = NWParameters.tcp
    p.includePeerToPeer = true
    return p
}()

func sendFramed(_ connection: NWConnection, _ data: Data, completion: @escaping (NWError?) -> Void) {
    var length = UInt32(data.count).littleEndian
    var frame = Data(bytes: &length, count: 4)
    frame.append(data)
    connection.send(content: frame, completion: .contentProcessed(completion))
}

class FrameReader {
    private var buffer = Data()
    private let connection: NWConnection
    private let handler: (Data) -> Void

    init(_ connection: NWConnection, handler: @escaping (Data) -> Void) {
        self.connection = connection
        self.handler = handler
        readMore()
    }

    private func readMore() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }
            if let data = data, !data.isEmpty {
                self.buffer.append(data)
                self.drainFrames()
            }
            if !isComplete && error == nil {
                self.readMore()
            }
        }
    }

    private func drainFrames() {
        while buffer.count >= 4 {
            let length = Int(buffer.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian })
            guard buffer.count >= 4 + length else { return }
            let payload = buffer.subdata(in: 4..<(4 + length))
            buffer.removeSubrange(0..<(4 + length))
            handler(payload)
        }
    }
}

func readExactly(_ fd: Int32, _ count: Int) -> Data? {
    var buf = Data(count: count)
    var total = 0
    while total < count {
        let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress! + total, count - total) }
        if n <= 0 { return nil }
        total += n
    }
    return buf
}

func createUDS(_ path: String) -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { fputs("[p2p] Failed to create socket\n", stderr); exit(1) }
    unlink(path)

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        path.utf8CString.withUnsafeBufferPointer { src in
            memcpy(ptr, src.baseAddress!, min(src.count, MemoryLayout.size(ofValue: ptr.pointee)))
        }
    }

    let bindResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
    }
    guard bindResult == 0 else { fputs("[p2p] Bind failed: \(errno)\n", stderr); exit(1) }
    listen(fd, 1)
    return fd
}

func runDiscover(controlPath: String) {
    let queue = DispatchQueue(label: "grove.discover", qos: .userInitiated)
    let lock = NSLock()
    var known: [String: [String: String]] = [:]
    var pendingLines: [String] = []
    var clientFd: Int32 = -1

    func sendLine(_ line: String) {
        lock.lock()
        if clientFd >= 0 {
            lock.unlock()
            let data = (line + "\n").data(using: .utf8)!
            data.withUnsafeBytes { ptr in
                var sent = 0
                while sent < data.count {
                    let n = write(clientFd, ptr.baseAddress! + sent, data.count - sent)
                    if n <= 0 { break }
                    sent += n
                }
            }
        } else {
            pendingLines.append(line)
            lock.unlock()
        }
    }

    let browser = NWBrowser(for: .bonjourWithTXTRecord(type: serviceType, domain: nil), using: p2pParams)

    browser.browseResultsChangedHandler = { results, changes in
        for change in changes {
            switch change {
            case .added(let result):
                if case .service(let name, _, _, _) = result.endpoint,
                   name.hasPrefix("grove-") {

                    if case .bonjour(let record) = result.metadata {
                        let dict = record.dictionary
                        var props: [String: String] = [:]
                        for key in ["role", "ws", "name", "uid", "script"] {
                            if let val = dict[key] {
                                props[key] = val
                            }
                        }
                        if props["role"] == "coordinator" {
                            let uid = props["uid"] ?? "?"
                            lock.lock()
                            let isNew = known[uid] == nil
                            known[uid] = props
                            lock.unlock()
                            if isNew {
                                let ws = props["ws"] ?? "2"
                                let cname = props["name"] ?? "?"
                                let script = props["script"] ?? "?"
                                sendLine("found \(cname) \(uid) \(ws) \(script)")
                                fputs("[discover] Found coordinator: \(cname) (\(uid)) ws=\(ws)\n", stderr)
                            }
                        }
                    }
                }
            case .removed(let result):
                if case .service(let name, _, _, _) = result.endpoint {
                    lock.lock()
                    for (uid, _) in known {
                        if name.contains(uid) {
                            known.removeValue(forKey: uid)
                            lock.unlock()
                            sendLine("lost \(uid)")
                            fputs("[discover] Lost: \(uid)\n", stderr)
                            return
                        }
                    }
                    lock.unlock()
                }
            default:
                break
            }
        }
    }

    browser.stateUpdateHandler = { state in
        if case .failed(let error) = state {
            fputs("[discover] Browser failed: \(error)\n", stderr)
        }
        if case .ready = state {
            fputs("[discover] Browser ready\n", stderr)
        }
    }

    browser.start(queue: queue)
    fputs("[discover] Browsing for coordinators over AWDL...\n", stderr)

    let serverFd = createUDS(controlPath)
    let fd = accept(serverFd, nil, nil)
    guard fd >= 0 else { fputs("[p2p] Accept failed\n", stderr); exit(1) }

    lock.lock()
    clientFd = fd
    let pending = pendingLines
    pendingLines = []
    lock.unlock()

    let readyMsg = "ready\n"
    _ = readyMsg.withCString { write(fd, $0, readyMsg.count) }
    for line in pending {
        let data = (line + "\n").data(using: .utf8)!
        data.withUnsafeBytes { ptr in
            var sent = 0
            while sent < data.count {
                let n = write(fd, ptr.baseAddress! + sent, data.count - sent)
                if n <= 0 { break }
                sent += n
            }
        }
    }

    fputs("[discover] Python connected. \(pending.count) buffered results sent.\n", stderr)
    dispatchMain()
}

func runMesh(cluster: String, worldSize: Int, controlPath: String, isCoordinator: Bool,
             clusterName: String, uid: String, scriptName: String) {
    let queue = DispatchQueue(label: "grove.p2p", qos: .userInitiated)
    let lock = NSLock()

    let rankPrefix = isCoordinator ? "0" : "1"
    let myName = "grove-\(cluster)-\(rankPrefix)-\(ProcessInfo.processInfo.processIdentifier)"
    var discoveredNames: Set<String> = [myName]
    var localSocket: Int32 = -1
    var myRank = -1
    var allNames: [String] = []

    let meshServerFd = createUDS(controlPath)
    fputs("[p2p] UDS socket created, accepting Python connection...\n", stderr)
    let meshClientFd = accept(meshServerFd, nil, nil)
    guard meshClientFd >= 0 else { fputs("[p2p] Accept failed\n", stderr); exit(1) }
    localSocket = meshClientFd
    fputs("[p2p] Python connected to UDS.\n", stderr)

    let listener = try! NWListener(using: p2pParams)

    if isCoordinator {
        var txt = NWTXTRecord()
        txt["role"] = "coordinator"
        txt["ws"] = String(worldSize)
        txt["name"] = clusterName
        txt["uid"] = uid
        txt["script"] = scriptName
        listener.service = NWListener.Service(name: myName, type: serviceType)
        listener.service?.txtRecordObject = txt
    } else {
        listener.service = NWListener.Service(name: myName, type: serviceType)
    }

    var connByRank: [Int: NWConnection] = [:]
    let allConnected = DispatchSemaphore(value: 0)
    let rankAssigned = DispatchSemaphore(value: 0)
    var connectedCount = 0

    var frameReaders: [FrameReader] = []

    func notifyDisconnect(_ rank: Int) {
        guard localSocket >= 0 else { return }
        var header = Data(count: 9)
        header[0] = OP_DISCONNECT
        header.withUnsafeMutableBytes {
            $0.storeBytes(of: UInt32(rank).littleEndian, toByteOffset: 1, as: UInt32.self)
            $0.storeBytes(of: UInt32(0).littleEndian, toByteOffset: 5, as: UInt32.self)
        }
        header.withUnsafeBytes { ptr in
            var sent = 0
            while sent < header.count {
                let n = write(localSocket, ptr.baseAddress! + sent, header.count - sent)
                if n <= 0 { break }
                sent += n
            }
        }
        fputs("[p2p] Notified Python: rank \(rank) disconnected\n", stderr)
    }

    func startReading(_ conn: NWConnection, _ rank: Int) {
        fputs("[p2p] startReading for rank \(rank) on conn \(conn.endpoint) state=\(conn.state)\n", stderr)
        let reader = FrameReader(conn) { data in
            fputs("[p2p] FrameReader got \(data.count) bytes from rank \(rank)\n", stderr)
            guard localSocket >= 0 else { return }
            var header = Data(count: 9)
            header[0] = OP_RECV
            header.withUnsafeMutableBytes {
                $0.storeBytes(of: UInt32(rank).littleEndian, toByteOffset: 1, as: UInt32.self)
                $0.storeBytes(of: UInt32(data.count).littleEndian, toByteOffset: 5, as: UInt32.self)
            }
            let combined = header + data
            combined.withUnsafeBytes { ptr in
                var sent = 0
                while sent < combined.count {
                    let n = write(localSocket, ptr.baseAddress! + sent, combined.count - sent)
                    if n <= 0 { break }
                    sent += n
                }
            }
        }
        frameReaders.append(reader)
    }

    func registerConn(_ rank: Int, _ conn: NWConnection) {
        lock.lock()
        connByRank[rank] = conn
        connectedCount += 1
        let done = connectedCount >= worldSize - 1
        lock.unlock()
        fputs("[p2p] Connection to rank \(rank) ready [\(connectedCount)/\(worldSize - 1)]\n", stderr)
        if done { allConnected.signal() }
    }

    listener.newConnectionHandler = { connection in
        connection.stateUpdateHandler = { state in
            if case .ready = state {
                fputs("[p2p] Incoming connection from \(connection.endpoint)\n", stderr)
                connection.receive(minimumIncompleteLength: 4, maximumLength: 4) { data, _, _, _ in
                    guard let data = data, data.count == 4 else { return }
                    let rank = Int(data.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian })
                    fputs("[p2p] Incoming handshake from rank \(rank)\n", stderr)

                    // Wait for rank assignment on background queue (can't block NW queue)
                    DispatchQueue.global().async {
                        rankAssigned.wait()
                        rankAssigned.signal()  // Re-signal for other waiters

                        var reply = Data(count: 4)
                        reply.withUnsafeMutableBytes { $0.storeBytes(of: UInt32(myRank).littleEndian, as: UInt32.self) }
                        connection.send(content: reply, completion: .contentProcessed { error in
                            if let error = error {
                                fputs("[p2p] Failed to send handshake reply: \(error)\n", stderr)
                            } else {
                                fputs("[p2p] Sent handshake reply to rank \(rank)\n", stderr)
                                // Incoming connections are for RECEIVING only
                                startReading(connection, rank)
                            }
                        })
                    }
                }
            }
            if case .failed(let error) = state {
                fputs("[p2p] Incoming connection failed: \(error)\n", stderr)
            }
            if case .cancelled = state {
                fputs("[p2p] Incoming connection cancelled\n", stderr)
            }
        }
        connection.start(queue: queue)
    }

    listener.stateUpdateHandler = { state in
        if case .failed(let error) = state {
            fputs("[p2p] Listener failed: \(error)\n", stderr)
            exit(1)
        }
    }

    listener.start(queue: queue)

    let browser = NWBrowser(for: .bonjourWithTXTRecord(type: serviceType, domain: nil), using: p2pParams)
    let discoveryDone = DispatchSemaphore(value: 0)

    browser.browseResultsChangedHandler = { results, changes in
        for change in changes {
            if case .added(let result) = change,
               case .service(let name, _, _, _) = result.endpoint,
               name.hasPrefix("grove-\(cluster)-") {

                lock.lock()
                let isNew = !discoveredNames.contains(name)
                if isNew { discoveredNames.insert(name) }
                let count = discoveredNames.count
                lock.unlock()

                if isNew {
                    fputs("[p2p] Discovered: \(name) [\(count)/\(worldSize)]\n", stderr)
                    if count >= worldSize { discoveryDone.signal() }
                }
            }
        }
    }

    browser.stateUpdateHandler = { state in
        if case .failed(let error) = state {
            fputs("[p2p] Browser failed: \(error)\n", stderr)
        }
    }

    browser.start(queue: queue)

    fputs("[p2p] Starting mesh: cluster=\(cluster), world_size=\(worldSize), coordinator=\(isCoordinator)\n", stderr)

    discoveryDone.wait()

    lock.lock()
    allNames = Array(discoveredNames).sorted()
    myRank = allNames.firstIndex(of: myName)!
    lock.unlock()

    rankAssigned.signal()
    fputs("[p2p] Rank assignment: \(myRank)/\(worldSize)\n", stderr)

    for result in browser.browseResults {
        if case .service(let name, _, _, _) = result.endpoint,
           name != myName,
           let rank = allNames.firstIndex(of: name) {

            fputs("[p2p] Connecting outgoing to rank \(rank)...\n", stderr)
            let conn = NWConnection(to: result.endpoint, using: p2pParams)
            conn.betterPathUpdateHandler = { available in
                if available {
                    fputs("[p2p] Better path available for rank \(rank)\n", stderr)
                }
            }
            conn.pathUpdateHandler = { path in
                fputs("[p2p] Rank \(rank) path: \(path.localEndpoint?.interface?.name ?? "?") -> \(path.remoteEndpoint?.interface?.name ?? "?")\n", stderr)
            }
            conn.stateUpdateHandler = { state in
                if case .ready = state {
                    if let path = conn.currentPath {
                        fputs("[p2p] Rank \(rank) connected via \(path.localEndpoint?.debugDescription ?? "?") iface=\(path.availableInterfaces.map { $0.name })\n", stderr)
                    }
                    var rankData = Data(count: 4)
                    rankData.withUnsafeMutableBytes { $0.storeBytes(of: UInt32(myRank).littleEndian, as: UInt32.self) }
                    conn.send(content: rankData, completion: .contentProcessed { _ in
                        fputs("[p2p] Sent handshake to rank \(rank)\n", stderr)
                        conn.receive(minimumIncompleteLength: 4, maximumLength: 4) { data, _, _, _ in
                            guard let data = data, data.count == 4 else { return }
                            let remoteRank = Int(data.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian })
                            fputs("[p2p] Got handshake reply from rank \(remoteRank)\n", stderr)
                            registerConn(rank, conn)
                        }
                    })
                }
                if case .failed(let error) = state {
                    fputs("[p2p] Outgoing to rank \(rank) failed: \(error)\n", stderr)
                    notifyDisconnect(rank)
                }
            }
            conn.start(queue: queue)
        }
    }

    allConnected.wait()

    fputs("[p2p] All connections established.\n", stderr)

    let readyMsg = "ready \(myRank)\n"
    _ = readyMsg.withCString { write(meshClientFd, $0, readyMsg.count) }
    fputs("[p2p] Sent ready to Python.\n", stderr)

    DispatchQueue.global(qos: .userInitiated).async {
        while true {
            guard let header = readExactly(meshClientFd, 9) else {
                fputs("[p2p] Python disconnected\n", stderr)
                exit(0)
            }
            let op = header[0]
            let peerRank = Int(header.withUnsafeBytes { $0.load(fromByteOffset: 1, as: UInt32.self).littleEndian })
            let length = Int(header.withUnsafeBytes { $0.load(fromByteOffset: 5, as: UInt32.self).littleEndian })

            if op == OP_SEND && length > 0 {
                guard let payload = readExactly(meshClientFd, length) else {
                    fputs("[p2p] Failed to read payload\n", stderr)
                    exit(1)
                }

                lock.lock()
                let conn = connByRank[peerRank]
                lock.unlock()

                if let conn = conn {
                    sendFramed(conn, payload) { error in
                        if let error = error {
                            fputs("[p2p] Send to rank \(peerRank) failed: \(error)\n", stderr)
                        }
                    }
                } else {
                    fputs("[p2p] No outgoing connection for rank \(peerRank)\n", stderr)
                }
            }
        }
    }

    dispatchMain()
}

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: grove-p2p-helper <discover|mesh> ...\n", stderr)
    exit(1)
}

let mode = CommandLine.arguments[1]

switch mode {
case "discover":
    guard CommandLine.arguments.count == 3 else {
        fputs("Usage: grove-p2p-helper discover <control_socket_path>\n", stderr)
        exit(1)
    }
    runDiscover(controlPath: CommandLine.arguments[2])

case "mesh":
    guard CommandLine.arguments.count >= 5 else {
        fputs("Usage: grove-p2p-helper mesh <cluster> <world_size> <control_socket_path> [--coordinator <name> <uid> <script>]\n", stderr)
        exit(1)
    }
    let cluster = CommandLine.arguments[2]
    let worldSize = Int(CommandLine.arguments[3])!
    let controlPath = CommandLine.arguments[4]

    var isCoord = false
    var cName = ""
    var cUid = ""
    var cScript = ""
    if CommandLine.arguments.count >= 9 && CommandLine.arguments[5] == "--coordinator" {
        isCoord = true
        cName = CommandLine.arguments[6]
        cUid = CommandLine.arguments[7]
        cScript = CommandLine.arguments[8]
    }

    runMesh(cluster: cluster, worldSize: worldSize, controlPath: controlPath,
            isCoordinator: isCoord, clusterName: cName, uid: cUid, scriptName: cScript)

default:
    fputs("Unknown mode: \(mode). Use 'discover' or 'mesh'.\n", stderr)
    exit(1)
}
