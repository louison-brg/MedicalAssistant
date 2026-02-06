import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel: ChatViewModel
    @State private var preferGPU: Bool = true

    init(viewModel: ChatViewModel = ChatViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 8) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                        }
                    }
                    .padding(.vertical)
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    withAnimation {
                        proxy.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                    }
                }
            }

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundColor(.red)
                    .padding(.horizontal)
            }

            HStack {
                Toggle(isOn: $preferGPU) {
                    Text(preferGPU ? "Mode: GPU (fallback)" : "Mode: CPU only")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
                .onChange(of: preferGPU) { _, newValue in
                    viewModel.setPerformance(preferGPU: newValue)
                }
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top, 6)

            if viewModel.isGenerating {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Generating response…")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 6)
            }

            Divider()

            HStack {
                TextField("Écrivez votre message…", text: $viewModel.currentInput)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .frame(minHeight: 40)
                    .disabled(viewModel.isGenerating)

                Button(action: {
                    if viewModel.isGenerating {
                        viewModel.cancelGeneration()
                    } else {
                        viewModel.sendMessage()
                    }
                }) {
                    Image(systemName: viewModel.isGenerating ? "stop.fill" : "paperplane.fill")
                        .foregroundColor(.white)
                        .padding(10)
                        .background(viewModel.isGenerating ? Color.red : Color.blue)
                        .clipShape(Circle())
                }
            }
            .padding()
            .background(Color(.systemGray6))

            Text("Not medical advice. For emergencies call local services.")
                .font(.caption2)
                .foregroundColor(.secondary)
                .padding(.bottom, 6)
        }
        .background(Color(.systemBackground))
        .onTapGesture { hideKeyboard() }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("MedLLM")
                    .font(.headline)
                Text("On-device medical assistant")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
            Button("Clear") {
                viewModel.clearMessages()
            }
            .font(.footnote)
        }
        .padding()
    }
}

#if canImport(UIKit)
extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil, from: nil, for: nil)
    }
}
#endif

#Preview {
    ContentView(viewModel: ChatViewModel(previewMode: true))
}
