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

    private init() throws {
        log.info("Initializing InferenceServer")

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        self.model = try gpt2_rank1(configuration: config)

        log.info("Core ML model loaded successfully")
    }

    func start() throws {
        log.info("Starting TCP listener on port \(self.port.rawValue)")

        listener = try NWListener(using: .tcp, on: port)

        listener?.newConnectionHandler = { [weak self] connection in
            guard let self = self else { return }

            let endpoint = "\(String(describing: connection.endpoint))"
            self.log.info("New connection from \(endpoint)")

            connection.stateUpdateHandler = { state in
                self.log.info("Connection state changed: \(String(describing: state))")
            }

            connection.start(queue: .global())
            self.receive(from: connection)
        }

        listener?.start(queue: .main)
        log.info("Listening on port \(self.port.rawValue)")
    }

    private func receive(from connection: NWConnection) {
        connection.receive(
            minimumIncompleteLength: 1,
            maximumLength: 1024 * 1024
        ) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }

            if let error = error {
                self.log.error("Receive error: \(error.localizedDescription)")
                connection.cancel()
                return
            }

            guard let data = data else {
                self.log.warning("Received empty data")
                return
            }

            self.log.info("Received \(data.count) bytes")

            // Sanity check
            let expectedBytes = 1 * 1 * 768 * MemoryLayout<Float>.size
            if data.count != expectedBytes {
                self.log.warning("Unexpected byte count: \(data.count), expected \(expectedBytes)")
            }

            let input = self.deserializeFP32(data)

            let start = CFAbsoluteTimeGetCurrent()
            let output = self.runModel(input)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

            let elapsedMs = String(format: "%.2f", elapsed)
            self.log.info("Inference completed in \(elapsedMs) ms")

            let response = self.serializeFP32(output)
            self.log.info("Sending \(response.count) bytes back")

            connection.send(content: response, completion: .contentProcessed { error in
                if let error = error {
                    self.log.error("Send error: \(error.localizedDescription)")
                } else {
                    self.log.info("Response sent successfully")
                }
                connection.cancel()
                self.log.info("Connection closed")
            })
        }
    }

    private func runModel(_ x: MLMultiArray) -> MLMultiArray {
        log.info("Running Core ML model")

        let input = gpt2_rank1Input(x: x)
        let output = try! model.prediction(input: input)

        log.info("Core ML prediction finished")
        return output.x
    }

    private func deserializeFP32(_ data: Data) -> MLMultiArray {
        log.info("Deserializing input tensor")

        let array = try! MLMultiArray(
            shape: [1, 1, 768],
            dataType: .float32
        )

        _ = data.withUnsafeBytes { src in
            memcpy(array.dataPointer, src.baseAddress!, data.count)
        }

        return array
    }

    private func serializeFP32(_ array: MLMultiArray) -> Data {
        log.info("Serializing output tensor")

        let size = array.count * MemoryLayout<Float>.size
        return Data(bytes: array.dataPointer, count: size)
    }
}
