import SwiftUI

struct ContentView: View {
    @State private var status = "Starting…"

    var body: some View {
        VStack(spacing: 20) {
            Text("GPT-2 iPad Inference Node")
                .font(.title)
            Text(status)
                .font(.body)
        }
        .onAppear {
            do {
                let server = try InferenceServer.shared
                try server.start()
                status = "Listening on port 8000"
            } catch {
                status = "Failed to start server"
            }
        }
    }
}
