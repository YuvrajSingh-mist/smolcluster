//
//  InferenceServer.swift
//  GPT2Node
//
//  Created by Yuvraj Singh on 29/01/26.
//


import Foundation
import CoreML
import Network

final class InferenceServer {

    static let shared = try! InferenceServer()

    private let port: NWEndpoint.Port = 8000
    private var listener: NWListener?
    private let model: gpt2_rank1

    private init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        self.model = try gpt2_rank1(configuration: config)
    }

    func start() throws {
        listener = try NWListener(using: .tcp, on: port)
        listener?.newConnectionHandler = { connection in
            connection.start(queue: .global())
            self.receive(from: connection)
        }
        listener?.start(queue: .main)
    }

    private func receive(from connection: NWConnection) {
        connection.receive(
            minimumIncompleteLength: 1,
            maximumLength: 1024 * 1024
        ) { data, _, _, _ in
            guard let data = data else { return }

            let input = self.deserializeFP32(data)
            let output = self.runModel(input)
            let response = self.serializeFP32(output)

            connection.send(content: response, completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }

    private func runModel(_ x: MLMultiArray) -> MLMultiArray {
        let input = gpt2_rank1Input(x: x)
        let output = try! model.prediction(input: input)
        return output.x
    }

    // 🔴 MOVE THESE HERE
    private func deserializeFP32(_ data: Data) -> MLMultiArray {
        let array = try! MLMultiArray(
            shape: [1, 1, 768],
            dataType: .float32
        )

        data.withUnsafeBytes { src in
            memcpy(array.dataPointer, src.baseAddress!, data.count)
        }

        return array
    }

    private func serializeFP32(_ array: MLMultiArray) -> Data {
        let size = array.count * MemoryLayout<Float>.size
        return Data(bytes: array.dataPointer, count: size)
    }
}
