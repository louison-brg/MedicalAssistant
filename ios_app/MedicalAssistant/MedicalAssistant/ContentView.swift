import SwiftUI
import LocalAuthentication

struct ContentView: View {
    @Environment(\.scenePhase) var scenePhase
    @StateObject private var viewModel: ChatViewModel
    @State private var preferGPU: Bool = true
    @FocusState private var inputFocused: Bool
    @State private var isUnlocked: Bool = false
    @State private var showingHistory: Bool = false

    init() {
        _viewModel = StateObject(wrappedValue: ChatViewModel())
    }

    init(viewModel: ChatViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        Group {
            if isUnlocked {
                mainChatView
            } else {
                lockedView
            }
        }
        .onAppear(perform: authenticate)
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                viewModel.unloadModel()
                isUnlocked = false
            } else if newPhase == .active && !isUnlocked {
                authenticate()
            }
        }
    }

    private var mainChatView: some View {
        VStack(spacing: 0) {
            header

            // Messages or empty state
            if viewModel.messages.isEmpty {
                emptyState
            } else {
                messageList
            }

            // Error banner
            if let error = viewModel.errorMessage {
                errorBanner(error)
            }

            inputBar
        }
        .background(backgroundView.ignoresSafeArea())
        .onTapGesture { inputFocused = false }
    }

    // MARK: - Authentication
    private func authenticate() {
        let context = LAContext()
        var error: NSError?

        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) {
            let reason = "Unlock MedLLM to access your medical conversations."
            context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) { success, authenticationError in
                DispatchQueue.main.async {
                    if success {
                        self.isUnlocked = true
                    } else {
                        // Fallback or retry
                    }
                }
            }
        } else {
            // No biometrics available, unlock by default or handle passcode
            self.isUnlocked = true
        }
    }

    private var lockedView: some View {
        VStack(spacing: 20) {
            Image(systemName: "lock.fill")
                .font(.system(size: 50))
                .foregroundColor(ChatTheme.accent)
            Text("MedLLM is Locked")
                .font(.title2.bold())
                .foregroundColor(.white)
            Button("Unlock") {
                authenticate()
            }
            .padding()
            .background(ChatTheme.userGradient)
            .clipShape(Capsule())
            .foregroundColor(.white)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(backgroundView.ignoresSafeArea())
    }

    // MARK: - Background

    private var backgroundView: some View {
        ZStack {
            // Base dark color
            Color(red: 0.06, green: 0.06, blue: 0.10)

            // Top-left colored blob
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(hue: 0.72, saturation: 0.50, brightness: 0.30),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 300
                    )
                )
                .frame(width: 500, height: 500)
                .offset(x: -150, y: -250)
                .blur(radius: 80)

            // Bottom-right colored blob
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(hue: 0.58, saturation: 0.35, brightness: 0.20),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 250
                    )
                )
                .frame(width: 400, height: 400)
                .offset(x: 120, y: 350)
                .blur(radius: 70)

            // Subtle noise texture overlay
            Color.white.opacity(0.015)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            Button {
                showingHistory = true
            } label: {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .padding(.trailing, 4)

            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.sessions.first(where: { $0.id == viewModel.activeSessionId })?.title ?? "MedLLM")
                    .font(ChatTheme.headerTitle)
                    .foregroundStyle(.white)
                    .lineLimit(1)
                Text("On-device medical assistant")
                    .font(ChatTheme.headerSubtitle)
                    .foregroundStyle(.white.opacity(0.5))
            }

            Spacer()

            gpuToggle

            Button {
                withAnimation(.spring(response: 0.35)) {
                    viewModel.createNewSession()
                }
            } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(.white)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(
            Rectangle()
                .fill(.ultraThinMaterial.opacity(0.5))
                .overlay(
                    Rectangle()
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.06), Color.clear],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                )
                .ignoresSafeArea(.container, edges: .top)
        )
        .sheet(isPresented: $showingHistory) {
            ChatHistoryView(viewModel: viewModel)
        }
    }

    private var gpuToggle: some View {
        Button {
            preferGPU.toggle()
            viewModel.setPerformance(preferGPU: preferGPU)
        } label: {
            HStack(spacing: 5) {
                Image(systemName: preferGPU ? "bolt.fill" : "cpu")
                    .font(.system(size: 10, weight: .bold))
                Text(preferGPU ? "GPU" : "CPU")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(preferGPU
                          ? AnyShapeStyle(ChatTheme.userGradient)
                          : AnyShapeStyle(Color.white.opacity(0.12)))
            )
            .shadow(color: preferGPU ? ChatTheme.accent.opacity(0.3) : .clear, radius: 6, y: 2)
        }
        .buttonStyle(.plain)
        .animation(.spring(response: 0.3), value: preferGPU)
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 20) {
            Spacer()

            // Glowing icon
            ZStack {
                Circle()
                    .fill(ChatTheme.accent.opacity(0.1))
                    .frame(width: 100, height: 100)
                    .blur(radius: 20)

                Image(systemName: "stethoscope")
                    .font(.system(size: 46, weight: .light))
                    .foregroundStyle(ChatTheme.accent.opacity(0.7))
            }

            Text("How can I help?")
                .font(ChatTheme.emptyStateTitle)
                .foregroundStyle(.white.opacity(0.85))

            Text("Ask any medical question.\nResponses are generated entirely on your device.")
                .font(ChatTheme.emptyStateBody)
                .foregroundStyle(.white.opacity(0.4))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)

            // Suggestion chips
            VStack(spacing: 10) {
                suggestionChip("What causes persistent headaches?")
                suggestionChip("Symptoms of vitamin D deficiency")
                suggestionChip("How to lower blood pressure naturally")
            }
            .padding(.top, 8)

            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func suggestionChip(_ text: String) -> some View {
        Button {
            viewModel.currentInput = text
            viewModel.sendMessage()
        } label: {
            Text(text)
                .font(.system(size: 14, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.65))
                .padding(.horizontal, 18)
                .padding(.vertical, 10)
                .background(
                    Capsule()
                        .fill(Color.white.opacity(0.07))
                        .overlay(
                            Capsule()
                                .strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5)
                        )
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Message List

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(
                            message: message,
                            onEdit: { newText in
                                viewModel.editMessage(id: message.id, newText: newText)
                            },
                            onRegenerate: {
                                viewModel.regenerate(from: message.id)
                            }
                        )
                            .id(message.id)
                    }
                }
                .padding(.vertical, 12)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: viewModel.messages.count) { _, _ in
                withAnimation(.easeOut(duration: 0.3)) {
                    proxy.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                }
            }
            .onChange(of: viewModel.messages.last?.text) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                }
            }
        }
    }

    // MARK: - Error Banner

    private func errorBanner(_ text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.caption)
            Text(text)
                .font(.caption)
        }
        .foregroundStyle(.red)
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(Color.red.opacity(0.12), in: Capsule())
        .padding(.horizontal)
        .padding(.bottom, 4)
        .transition(.move(edge: .top).combined(with: .opacity))
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 0) {
            // Subtle top edge line
            Rectangle()
                .fill(Color.white.opacity(0.06))
                .frame(height: 0.5)

            HStack(alignment: .bottom, spacing: 10) {
                // Text input
                TextField("Ask a medical question…", text: $viewModel.currentInput, axis: .vertical)
                    .font(ChatTheme.inputFont)
                    .foregroundStyle(.white)
                    .lineLimit(1...5)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: ChatTheme.inputRadius)
                            .fill(Color.white.opacity(0.07))
                            .overlay(
                                RoundedRectangle(cornerRadius: ChatTheme.inputRadius)
                                    .strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5)
                            )
                    )
                    .focused($inputFocused)
                    .disabled(viewModel.isGenerating)
                    .submitLabel(.send)
                    .onSubmit { viewModel.sendMessage() }
                    .tint(ChatTheme.accent)

                // Send / Stop button
                Button {
                    if viewModel.isGenerating {
                        viewModel.cancelGeneration()
                    } else {
                        viewModel.sendMessage()
                        inputFocused = false
                    }
                } label: {
                    ZStack {
                        Circle()
                            .fill(buttonFill)
                            .frame(width: 42, height: 42)
                            .shadow(color: buttonShadow, radius: 6, y: 2)

                        Image(systemName: viewModel.isGenerating ? "stop.fill" : "arrow.up")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundStyle(.white)
                    }
                }
                .disabled(!canSend && !viewModel.isGenerating)
                .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.isGenerating)
                .animation(.spring(response: 0.3, dampingFraction: 0.7), value: canSend)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)

            // Disclaimer
            Text("⚕️ Not medical advice · 100% Local Inference · Data remains on device")
                .font(ChatTheme.disclaimerFont)
                .foregroundStyle(.white.opacity(0.35))
                .padding(.bottom, 6)
        }
        .background(
            Rectangle()
                .fill(Color(red: 0.06, green: 0.06, blue: 0.10).opacity(0.9))
                .overlay(
                    Rectangle()
                        .fill(.ultraThinMaterial.opacity(0.3))
                )
                .ignoresSafeArea(.container, edges: .bottom)
        )
    }

    private var buttonFill: AnyShapeStyle {
        if viewModel.isGenerating {
            return AnyShapeStyle(Color.red.opacity(0.85))
        } else if canSend {
            return AnyShapeStyle(ChatTheme.userGradient)
        } else {
            return AnyShapeStyle(Color.white.opacity(0.1))
        }
    }

    private var buttonShadow: Color {
        if viewModel.isGenerating {
            return .red.opacity(0.3)
        } else if canSend {
            return ChatTheme.accent.opacity(0.3)
        } else {
            return .clear
        }
    }

    private var canSend: Bool {
        !viewModel.currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

// MARK: - Keyboard Helper

#if canImport(UIKit)
extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil, from: nil, for: nil)
    }
}
#endif

// MARK: - Preview

#Preview {
    ContentView()
}
