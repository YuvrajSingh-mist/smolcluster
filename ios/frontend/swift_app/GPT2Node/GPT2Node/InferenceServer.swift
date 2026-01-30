import Foundation
import CoreML
import Network
import os.log

final class InferenceServer {

    static let shared = try! InferenceServer()
    private let log = Logger(subsystem: "com.yuvraj.GPT2Node", category: "Inference")

    private let port: NWEndpoint.Port = 8000
    private var listener: NWListener?
    private let model: gpt2_rank1

    // TCP framing state
    private var buffer = Data()
    private var expectedSeqLen: Int? = nil
    private var expectedPayloadLen: Int? = nil

    private init() throws {
        log.info("Initializing InferenceServer")

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        self.model = try gpt2_rank1(configuration: config)

        log.info("Core ML model loaded")
    }

    func start() throws {
        listener = try NWListener(using: .tcp, on: port)

        listener?.newConnectionHandler = { [weak self] connection in
            guard let self = self else { return }

            self.log.info("New connection")

            connection.stateUpdateHandler = { state in
                self.log.info("Connection state: \(String(describing: state))")
            }

            connection.start(queue: .global(qos: .userInitiated))
            self.receive(from: connection)
        }

        listener?.start(queue: .main)
        log.info("Listening on port \(self.port.rawValue)")
    }

    // MARK: - Receive Loop

    private func receive(from connection: NWConnection) {
        connection.receive(
            minimumIncompleteLength: 1,
            maximumLength: 64 * 1024
        ) { [weak self] data, _, _, error in
            guard let self = self else { return }

            if let error = error {
                self.log.error("Receive error: \(error.localizedDescription)")
                connection.cancel()
                return
            }

            guard let data = data else {
                self.receive(from: connection)
                return
            }

            self.buffer.append(data)
            self.processBuffer(connection)
            self.receive(from: connection)
        }
    }

    // MARK: - Framing

    private func processBuffer(_ connection: NWConnection) {
        while true {

            // ---- read seq_len (4 bytes) ----
            if expectedSeqLen == nil {
                guard buffer.count >= 4 else { return }

                var v: UInt32 = 0
                _ = withUnsafeMutableBytes(of: &v) { dst in
                    buffer.copyBytes(to: dst, count: 4)
                }

                expectedSeqLen = Int(UInt32(bigEndian: v))
                buffer.removeFirst(4)
            }

            // ---- read payload length (4 bytes) ----
            if expectedPayloadLen == nil {
                guard buffer.count >= 4 else { return }

                var v: UInt32 = 0
                _ = withUnsafeMutableBytes(of: &v) { dst in
                    buffer.copyBytes(to: dst, count: 4)
                }

                expectedPayloadLen = Int(UInt32(bigEndian: v))
                buffer.removeFirst(4)
            }

            // ---- read payload ----
            guard
                let seqLen = expectedSeqLen,
                let payloadLen = expectedPayloadLen,
                buffer.count >= payloadLen
            else { return }

            let payload = buffer.prefix(payloadLen)
            buffer.removeFirst(payloadLen)

            expectedSeqLen = nil
            expectedPayloadLen = nil

            handleTensor(Data(payload), seqLen, connection)
        }
    }

    // MARK: - Inference

    private func handleTensor(
        _ data: Data,
        _ seqLen: Int,
        _ connection: NWConnection
    ) {
        let input = deserializeFP32(data, seqLen: seqLen)

        // GPU inference must run on main thread
        DispatchQueue.main.async {
            let start = CFAbsoluteTimeGetCurrent()
            let output = self.runModel(input)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

            self.log.info("Inference completed in \(String(format: "%.2f", elapsed)) ms")

            let response = self.serializeFP32(output)

            // ---- send back: seq_len + payload_len + payload ----
            var h1 = UInt32(seqLen).bigEndian
            var h2 = UInt32(response.count).bigEndian

            var out = Data()
            out.append(Data(bytes: &h1, count: 4))
            out.append(Data(bytes: &h2, count: 4))
            out.append(response)

            connection.send(content: out, completion: .contentProcessed { error in
                if let error = error {
                    self.log.error("Send error: \(error.localizedDescription)")
                    connection.cancel()
                }
            })
        }
    }

    // MARK: - Core ML

    private func runModel(_ x: MLMultiArray) -> MLMultiArray {
        let input = gpt2_rank1Input(x: x)
        let output = try! model.prediction(input: input)
        return output.x
    }

    // MARK: - Tensor helpers

    private func deserializeFP32(_ data: Data, seqLen: Int) -> MLMultiArray {
        let array = try! MLMultiArray(
            shape: [1, NSNumber(value: seqLen), 768],
            dataType: .float32
        )

        data.withUnsafeBytes { src in
            memcpy(array.dataPointer, src.baseAddress!, data.count)
        }

        return array
    }

    private func serializeFP32(_ array: MLMultiArray) -> Data {
        let count = array.count
        var out = Data(count: count * MemoryLayout<Float>.size)

        out.withUnsafeMutableBytes { dst in
            let ptr = dst.bindMemory(to: Float.self).baseAddress!
            for i in 0..<count {
                ptr[i] = array[i].floatValue
            }
        }

        return out
    }
}
