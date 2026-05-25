import SwiftUI
import LocalAuthentication

struct ContentView: View {
    @Environment(\.scenePhase) var scenePhase
    @StateObject private var viewModel: ChatViewModel
    @State private var preferGPU: Bool = true
    @FocusState private var inputFocused: Bool
    @State private var isUnlocked: Bool = false
    @State private var showingHistory: Bool = false
    @State private var scrollPosition: UUID?
    @Namespace private var composerNamespace

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
        .onChange(of: viewModel.lastSubmittedUserMessageID) { _, submittedID in
            guard submittedID != nil else { return }
            Task {
                try? await Task.sleep(nanoseconds: 550_000_000)
                await MainActor.run {
                    viewModel.clearComposerMorph()
                }
            }
        }
    }

    private var accentColor: Color {
        ChatTheme.electricBlue
    }

    private var accentGradient: LinearGradient {
        ChatTheme.accentGradient(load: 0)
    }

    private var mainChatView: some View {
        GlassEffectContainer(spacing: 26) {
            VStack(spacing: 0) {
                header

                if viewModel.messages.isEmpty {
                    emptyState
                } else {
                    messageList
                }

                if let error = viewModel.errorMessage {
                    errorBanner(error)
                }
            }
        }
        .background(backgroundView.ignoresSafeArea())
        .safeAreaInset(edge: .bottom, spacing: 0) {
            inputBar
        }
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
                .foregroundColor(accentColor)
            Text("MedLLM is Locked")
                .font(.title2.bold())
                .foregroundColor(.white)
            Button("Unlock") {
                authenticate()
            }
            .padding()
            .background(accentGradient)
            .clipShape(Capsule())
            .foregroundColor(.white)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(backgroundView.ignoresSafeArea())
    }

    // MARK: - Background

    private var backgroundView: some View {
        ZStack {
            // Rich MeshGradient for Liquid Glass refraction
            MeshGradient(
                width: 3,
                height: 3,
                points: [
                    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                    [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                    [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
                ],
                colors: [
                    Color(red: 0.04, green: 0.04, blue: 0.12),  // deep navy
                    Color(red: 0.06, green: 0.05, blue: 0.16),  // midnight indigo
                    Color(red: 0.05, green: 0.08, blue: 0.14),  // dark teal-navy

                    Color(red: 0.08, green: 0.06, blue: 0.18),  // deep violet
                    Color(red: 0.06, green: 0.06, blue: 0.10),  // center base
                    Color(red: 0.04, green: 0.10, blue: 0.16),  // teal accent

                    Color(red: 0.10, green: 0.08, blue: 0.22),  // violet glow
                    Color(red: 0.05, green: 0.07, blue: 0.14),  // deep blue
                    Color(red: 0.06, green: 0.12, blue: 0.18)   // teal edge
                ]
            )

            // Scroll-reactive diffuse glow – keeps the glass alive
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            accentColor.opacity(0.28),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 300
                    )
                )
                .frame(width: 500, height: 500)
                .offset(x: -150, y: -250)
                .blur(radius: 78)

            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            ChatTheme.deepViolet.opacity(0.18),
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

            Color.white.opacity(0.012)

            Rectangle()
                .fill(.ultraThinMaterial.opacity(0.04))
                .blur(radius: 18)
                .blendMode(.screen)
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
        .padding(.top, 14)
        .padding(.bottom, 12)
        .background(
            UnevenRoundedRectangle(
                topLeadingRadius: 0,
                bottomLeadingRadius: 26,
                bottomTrailingRadius: 26,
                topTrailingRadius: 0
            )
                .fill(.ultraThinMaterial)
                .overlay(
                    UnevenRoundedRectangle(
                        topLeadingRadius: 0,
                        bottomLeadingRadius: 26,
                        bottomTrailingRadius: 26,
                        topTrailingRadius: 0
                    )
                    .strokeBorder(ChatTheme.glassStroke, lineWidth: ChatTheme.glassLineWidth)
                )
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(accentColor.opacity(0.18))
                        .frame(height: 1)
                        .blur(radius: 3)
                }
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
                    .fill(preferGPU ? AnyShapeStyle(accentGradient) : AnyShapeStyle(.ultraThinMaterial))
                    .overlay(
                        Capsule()
                            .strokeBorder(ChatTheme.glassStroke, lineWidth: ChatTheme.glassLineWidth)
                    )
            )
            .shadow(color: preferGPU ? accentColor.opacity(0.3) : .clear, radius: 6, y: 2)
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
                    .fill(accentColor.opacity(0.12))
                    .frame(width: 100, height: 100)
                    .blur(radius: 20)

                Image(systemName: "stethoscope")
                    .font(.system(size: 46, weight: .light))
                    .foregroundStyle(accentColor.opacity(0.8))
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
                        .fill(.ultraThinMaterial)
                        .overlay(
                            Capsule()
                                .strokeBorder(ChatTheme.glassStroke, lineWidth: ChatTheme.glassLineWidth)
                        )
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Message List

    private var messageList: some View {
        ScrollView {
            LazyVStack(spacing: 4) {
                ForEach(viewModel.messages) { message in
                    MessageBubble(
                        message: message,
                        accentColor: accentColor,
                        morphNamespace: composerNamespace,
                        morphFromComposer: message.id == viewModel.lastSubmittedUserMessageID,
                        onEdit: { newText in
                            viewModel.editMessage(id: message.id, newText: newText)
                        },
                        onRegenerate: {
                            viewModel.regenerate(from: message.id)
                        }
                    )
                }
            }
            .padding(.vertical, 12)
            .scrollTargetLayout()
            .animation(.spring(response: 0.45, dampingFraction: 0.75), value: viewModel.messages.count)
        }
        .scrollPosition(id: $scrollPosition, anchor: .bottom)
        .defaultScrollAnchor(.bottom)
        .scrollDismissesKeyboard(.interactively)
        .onChange(of: viewModel.messages.count) { _, _ in
            withAnimation(.spring(response: 0.45, dampingFraction: 0.75)) {
                scrollPosition = viewModel.messages.last?.id
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
        VStack(spacing: 8) {
            // The floating pill
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Ask a medical question…", text: $viewModel.currentInput, axis: .vertical)
                    .font(.system(size: 16, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.96))
                    .lineLimit(1...5)
                    .focused($inputFocused)
                    .disabled(viewModel.isGenerating)
                    .submitLabel(.send)
                    .onSubmit {
                        viewModel.sendMessage()
                        inputFocused = false
                    }
                    .tint(accentColor)
                    .padding(.vertical, 14)
                    .padding(.leading, 18)

                sendButton
                    .padding(.trailing, 8)
                    .padding(.bottom, 6)
            }
            .background(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                            .strokeBorder(ChatTheme.glassStroke, lineWidth: ChatTheme.glassLineWidth)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                            .strokeBorder(accentColor.opacity(inputFocused ? 0.3 : 0.0), lineWidth: 0.8)
                    )
                    .shadow(color: Color.black.opacity(0.18), radius: 14, y: 6)
            )
            .padding(.horizontal, 14)

            // Disclaimer
            Text("⚕️ Not medical advice · 100% Local Inference · Data remains on device")
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.35))
                .padding(.bottom, 6)
        }
        .padding(.top, 4)
        // Stable frame for the composer morph
        .overlay(alignment: .bottomTrailing) {
            Color.clear
                .frame(width: 1, height: 1)
                .matchedGeometryEffect(id: "composerMorph", in: composerNamespace, properties: .frame, anchor: .bottomTrailing, isSource: true)
        }
    }

    private var sendButton: some View {
        Button {
            if viewModel.isGenerating {
                viewModel.cancelGeneration()
            } else {
                viewModel.sendMessage()
                inputFocused = false
            }
        } label: {
            Circle()
                .fill(buttonFill)
                .frame(width: 32, height: 32)
                .overlay(
                    Image(systemName: viewModel.isGenerating ? "stop.fill" : "arrow.up")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                )
                .shadow(color: buttonShadow, radius: 6, y: 2)
        }
        .buttonStyle(.plain)
        .disabled(!canSend && !viewModel.isGenerating)
        .animation(.easeInOut(duration: 0.2), value: viewModel.isGenerating)
        .animation(.easeInOut(duration: 0.2), value: canSend)
    }

    private var buttonFill: AnyShapeStyle {
        if viewModel.isGenerating {
            return AnyShapeStyle(Color.red.opacity(0.85))
        } else if canSend {
            return AnyShapeStyle(accentGradient)
        } else {
            return AnyShapeStyle(.ultraThinMaterial)
        }
    }

    private var buttonShadow: Color {
        if viewModel.isGenerating {
            return .red.opacity(0.3)
        } else if canSend {
            return accentColor.opacity(0.3)
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
