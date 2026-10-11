import LocalAuthentication
import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @StateObject private var viewModel: ChatViewModel

    @State private var isUnlocked = false
    @State private var isAuthenticating = false
    @State private var authenticationMessage: String?
    @State private var showingHistory = false
    @State private var showingMedicalInfo = false
    @State private var showingRenameDialog = false
    @State private var showingClearConfirmation = false
    @State private var renameText = ""
    @State private var scrollPosition: UUID?
    @State private var isScrolledUp = false

    init() {
        _viewModel = StateObject(wrappedValue: ChatViewModel())
    }

    init(viewModel: ChatViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        Group {
            if isUnlocked {
                chat
            } else {
                lockScreen
            }
        }
        .preferredColorScheme(.dark)
        .onAppear(perform: authenticate)
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                viewModel.unloadModel()
                isUnlocked = false
            case .active where !isUnlocked:
                authenticate()
            default:
                break
            }
        }
    }

    // MARK: - Main chat

    private var chat: some View {
        ZStack {
            ChatBackground()
                .ignoresSafeArea()

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
        .safeAreaInset(edge: .bottom, spacing: 0) {
            ChatComposer(
                isGenerating: viewModel.isGenerating,
                onSend: viewModel.sendMessage,
                onStop: viewModel.cancelGeneration
            )
        }
        .sheet(isPresented: $showingHistory) {
            ChatHistoryView(viewModel: viewModel)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
                .presentationBackground(.ultraThinMaterial)
        }
        .sheet(isPresented: $showingMedicalInfo) {
            MedicalDisclaimerView(isDismissable: true) {
                showingMedicalInfo = false
            }
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
            .presentationBackground(.ultraThinMaterial)
        }
        .alert("Renommer la discussion", isPresented: $showingRenameDialog) {
            TextField("Titre de la consultation", text: $renameText)
            Button("Annuler", role: .cancel) {}
            Button("Enregistrer") {
                if let id = viewModel.activeSessionId {
                    viewModel.renameSession(id: id, newTitle: renameText)
                }
            }
        }
        .confirmationDialog(
            "Effacer cette discussion ?",
            isPresented: $showingClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Effacer les messages", role: .destructive) {
                withAnimation {
                    viewModel.clearMessages()
                }
            }
            Button("Annuler", role: .cancel) {}
        } message: {
            Text("Tous les messages de la consultation active seront effacés.")
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            headerButton(systemName: "line.3.horizontal") {
                HapticsHelper.playLightImpact()
                showingHistory = true
            }
            .accessibilityLabel("Historique des conversations")

            // Brand avatar (tappable for disclaimer & emergency info)
            Button {
                HapticsHelper.playLightImpact()
                showingMedicalInfo = true
            } label: {
                ZStack {
                    Circle()
                        .fill(ChatTheme.heroCrossGradient)
                    Image(systemName: "cross.case.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .frame(width: 36, height: 36)
                .shadow(color: ChatTheme.auroraCyan.opacity(0.35), radius: 10, y: 2)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Informations et urgences médicales")

            // Title and offline indicator
            Button {
                renameText = activeSessionTitle
                showingRenameDialog = true
            } label: {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 4) {
                        Text(activeSessionTitle)
                            .font(ChatTheme.headerTitle)
                            .foregroundStyle(.white)
                            .lineLimit(1)
                        Image(systemName: "pencil")
                            .font(.system(size: 10))
                            .foregroundStyle(.white.opacity(0.35))
                    }

                    HStack(spacing: 5) {
                        Circle()
                            .fill(viewModel.isGenerating ? ChatTheme.auroraCyan : ChatTheme.emerald)
                            .frame(width: 5, height: 5)
                            .shadow(color: (viewModel.isGenerating ? ChatTheme.auroraCyan : ChatTheme.emerald).opacity(0.8), radius: 3)
                        Text(viewModel.isGenerating ? "Analyse en cours…" : "IA locale · Privé")
                            .font(ChatTheme.headerSubtitle)
                            .foregroundStyle(.white.opacity(0.55))
                    }
                }
            }
            .buttonStyle(.plain)

            Spacer(minLength: 6)

            Button {
                showingMedicalInfo = true
            } label: {
                privacyBadge
            }
            .buttonStyle(.plain)

            // More actions Menu
            Menu {
                if let active = viewModel.activeSession, !active.messages.isEmpty {
                    ShareLink(item: active.exportText()) {
                        Label("Partager la consultation", systemImage: "square.and.arrow.up")
                    }
                }

                Button {
                    renameText = activeSessionTitle
                    showingRenameDialog = true
                } label: {
                    Label("Renommer la discussion", systemImage: "pencil")
                }

                Button {
                    showingMedicalInfo = true
                } label: {
                    Label("Urgences & Avertissement", systemImage: "cross.case")
                }

                if !viewModel.messages.isEmpty {
                    Divider()
                    Button(role: .destructive) {
                        showingClearConfirmation = true
                    } label: {
                        Label("Effacer les messages", systemImage: "trash")
                    }
                }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.85))
                    .frame(width: 36, height: 36)
                    .background(
                        Circle()
                            .fill(.ultraThinMaterial)
                            .overlay(
                                Circle().strokeBorder(Color.white.opacity(0.14), lineWidth: 0.5)
                            )
                    )
                    .shadow(color: Color.black.opacity(0.15), radius: 4, y: 2)
            }

            headerButton(systemName: "square.and.pencil") {
                HapticsHelper.playLightImpact()
                withAnimation(reduceMotion ? nil : .snappy(duration: 0.28)) {
                    viewModel.createNewSession()
                }
            }
            .accessibilityLabel("Nouvelle conversation")
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 12)
        .background(.ultraThinMaterial)
        .overlay(alignment: .bottom) {
            LinearGradient(
                colors: [
                    .clear,
                    ChatTheme.auroraCyan.opacity(0.3),
                    ChatTheme.electricBlue.opacity(0.15),
                    .clear
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(height: 0.6)
        }
    }

    private var activeSessionTitle: String {
        viewModel.sessions.first(where: { $0.id == viewModel.activeSessionId })?.title ?? "MedLLM"
    }

    private func headerButton(systemName: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.white.opacity(0.88))
                .frame(width: 36, height: 36)
                .background(
                    Circle()
                        .fill(.ultraThinMaterial)
                        .overlay(
                            Circle()
                                .strokeBorder(Color.white.opacity(0.14), lineWidth: 0.5)
                        )
                )
                .shadow(color: Color.black.opacity(0.15), radius: 4, y: 2)
        }
        .buttonStyle(.plain)
    }

    private var privacyBadge: some View {
        HStack(spacing: 5) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(ChatTheme.mint)
            Text("100% LOCAL")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .tracking(0.6)
        }
        .foregroundStyle(.white.opacity(0.75))
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(
            Capsule()
                .fill(ChatTheme.mint.opacity(0.08))
                .overlay(
                    Capsule()
                        .strokeBorder(ChatTheme.mint.opacity(0.24), lineWidth: 0.6)
                )
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Inférence 100% locale sur cet appareil")
    }

    // MARK: - Empty state

    private var emptyState: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer(minLength: 28)

                // Hero glowing badge
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    ChatTheme.auroraCyan.opacity(0.22),
                                    ChatTheme.electricBlue.opacity(0.08),
                                    .clear
                                ],
                                center: .center,
                                startRadius: 20,
                                endRadius: 75
                            )
                        )
                        .frame(width: 140, height: 140)

                    Circle()
                        .strokeBorder(
                            LinearGradient(
                                colors: [
                                    Color.white.opacity(0.35),
                                    ChatTheme.auroraCyan.opacity(0.2),
                                    Color.clear
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                        .frame(width: 96, height: 96)

                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [
                                    ChatTheme.auroraCyan.opacity(0.18),
                                    ChatTheme.electricBlue.opacity(0.10)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 84, height: 84)

                    Image(systemName: "stethoscope")
                        .font(.system(size: 40, weight: .light))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(
                            LinearGradient(
                                colors: [ChatTheme.auroraCyan, ChatTheme.sky],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                }
                .shadow(color: ChatTheme.electricBlue.opacity(0.32), radius: 28)

                // Title & Subtitle
                VStack(spacing: 8) {
                    Text("Comment puis-je vous aider ?")
                        .font(ChatTheme.emptyStateTitle)
                        .foregroundStyle(.white.opacity(0.96))

                    Text("Posez vos questions sur vos symptômes, médicaments ou bilans.\nTous les calculs restent strictement privés sur cet appareil.")
                        .font(ChatTheme.emptyStateBody)
                        .foregroundStyle(.white.opacity(0.52))
                        .multilineTextAlignment(.center)
                        .lineSpacing(3.5)
                }
                .padding(.horizontal, 24)

                // Quick Prompt Cards
                VStack(spacing: 11) {
                    promptCard(
                        title: "Analyse de symptômes",
                        subtitle: "Maux de tête, fièvre, douleurs musculaires, fatigue",
                        icon: "stethoscope",
                        tint: ChatTheme.auroraCyan,
                        query: "J'aimerais analyser des symptômes récents."
                    )

                    promptCard(
                        title: "Guide des médicaments",
                        subtitle: "Indications, interactions et effets secondaires",
                        icon: "pills.fill",
                        tint: ChatTheme.deepViolet,
                        query: "Pouvez-vous m'expliquer le mode d'action d'un médicament et ses effets indésirables ?"
                    )

                    promptCard(
                        title: "Comprendre une analyse médicale",
                        subtitle: "Bilan sanguin, numération, constantes et glycémie",
                        icon: "drop.fill",
                        tint: ChatTheme.coral,
                        query: "Aidez-moi à comprendre les valeurs de mon dernier bilan sanguin."
                    )

                    promptCard(
                        title: "Préparer ma consultation",
                        subtitle: "Lister les questions ciblées pour mon médecin",
                        icon: "list.clipboard.fill",
                        tint: ChatTheme.emerald,
                        query: "Aidez-moi à préparer une liste de questions ciblées pour mon prochain rendez-vous médical."
                    )
                }
                .padding(.horizontal, 16)

                Spacer(minLength: 28)
            }
            .frame(maxWidth: 560)
            .frame(maxWidth: .infinity)
        }
        .scrollIndicators(.hidden)
    }

    private func promptCard(
        title: String,
        subtitle: String,
        icon: String,
        tint: Color,
        query: String
    ) -> some View {
        Button {
            HapticsHelper.playLightImpact()
            viewModel.sendMessage(query)
        } label: {
            HStack(spacing: 14) {
                // Tinted Icon Badge
                ZStack {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(tint.opacity(0.16))
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .strokeBorder(tint.opacity(0.3), lineWidth: 0.7)
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(tint)
                }
                .frame(width: 40, height: 40)

                // Texts
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white.opacity(0.92))

                    Text(subtitle)
                        .font(.system(size: 12, weight: .regular, design: .rounded))
                        .foregroundStyle(.white.opacity(0.48))
                        .lineLimit(1)
                }

                Spacer(minLength: 8)

                Image(systemName: "arrow.up.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.28))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.white.opacity(0.025))
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(0.16),
                                tint.opacity(0.12),
                                Color.white.opacity(0.04)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.6
                    )
            )
            .shadow(color: Color.black.opacity(0.15), radius: 8, y: 3)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Messages

    private var messageList: some View {
        ZStack(alignment: .bottomTrailing) {
            ScrollView {
                LazyVStack(spacing: 6) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(
                            message: message,
                            accentColor: ChatTheme.electricBlue,
                            onEdit: { viewModel.editMessage(id: message.id, newText: $0) },
                            onRegenerate: { viewModel.regenerate(from: message.id) }
                        )
                        .id(message.id)
                    }
                }
                .padding(.vertical, 14)
                .scrollTargetLayout()
            }
            .scrollPosition(id: $scrollPosition, anchor: .bottom)
            .defaultScrollAnchor(.bottom)
            .scrollDismissesKeyboard(.interactively)
            .scrollIndicators(.hidden)
            .onChange(of: scrollPosition) { _, newId in
                if let lastId = viewModel.messages.last?.id {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isScrolledUp = (newId != nil && newId != lastId)
                    }
                }
            }
            .onChange(of: viewModel.messages.count) { _, _ in
                scrollToLatest(animated: true)
                isScrolledUp = false
            }
            .onChange(of: viewModel.messages.last?.text.count) { _, _ in
                if !isScrolledUp {
                    scrollToLatest(animated: false)
                }
            }

            // Floating scroll to bottom button
            if isScrolledUp {
                Button {
                    HapticsHelper.playLightImpact()
                    scrollToLatest(animated: true)
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isScrolledUp = false
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 12, weight: .bold))
                        Text("Dernier message")
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .overlay(
                        Capsule().strokeBorder(ChatTheme.auroraCyan.opacity(0.4), lineWidth: 0.8)
                    )
                    .shadow(color: ChatTheme.electricBlue.opacity(0.35), radius: 8, y: 3)
                }
                .buttonStyle(.plain)
                .padding(.trailing, 16)
                .padding(.bottom, 12)
                .transition(.scale.combined(with: .opacity))
            }
        }
    }

    private func scrollToLatest(animated: Bool) {
        guard let id = viewModel.messages.last?.id else { return }
        if animated && !reduceMotion {
            withAnimation(.snappy(duration: 0.28)) { scrollPosition = id }
        } else {
            scrollPosition = id
        }
    }

    private func errorBanner(_ text: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(ChatTheme.amber)
            Text(text)
                .lineLimit(2)
        }
        .font(.caption)
        .foregroundStyle(ChatTheme.amber)
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(ChatTheme.amber.opacity(0.12), in: Capsule())
        .overlay(Capsule().strokeBorder(ChatTheme.amber.opacity(0.25), lineWidth: 0.5))
        .padding(.horizontal)
        .padding(.bottom, 6)
    }

    // MARK: - Authentication

    private func authenticate() {
        guard !isAuthenticating else { return }
        isAuthenticating = true
        authenticationMessage = nil

        let context = LAContext()
        context.localizedCancelTitle = "Plus tard"
        var policyError: NSError?
        guard context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &policyError) else {
            isAuthenticating = false
            isUnlocked = true
            return
        }

        context.evaluatePolicy(
            .deviceOwnerAuthentication,
            localizedReason: "Déverrouillez vos conversations médicales privées."
        ) { success, error in
            DispatchQueue.main.async {
                isAuthenticating = false
                isUnlocked = success
                if !success, let error {
                    authenticationMessage = error.localizedDescription
                }
            }
        }
    }

    private var lockScreen: some View {
        ZStack {
            ChatBackground()
                .ignoresSafeArea()

            VStack(spacing: 26) {
                // Biometric badge
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    ChatTheme.auroraCyan.opacity(0.25),
                                    ChatTheme.electricBlue.opacity(0.08),
                                    .clear
                                ],
                                center: .center,
                                startRadius: 15,
                                endRadius: 70
                            )
                        )
                        .frame(width: 130, height: 130)

                    Circle()
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.3), ChatTheme.auroraCyan.opacity(0.2), .clear],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                        .frame(width: 96, height: 96)

                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 42, weight: .light))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(ChatTheme.auroraCyan)
                }
                .shadow(color: ChatTheme.electricBlue.opacity(0.32), radius: 24)

                VStack(spacing: 7) {
                    Text("MedLLM est verrouillé")
                        .font(.system(.title2, design: .rounded, weight: .bold))
                        .foregroundStyle(.white)

                    Text("Vos consultations médicales sont chiffrées\net stockées exclusivement sur cet appareil.")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.55))
                        .multilineTextAlignment(.center)
                        .lineSpacing(3)
                }

                if let authenticationMessage {
                    Text(authenticationMessage)
                        .font(.caption)
                        .foregroundStyle(ChatTheme.amber)
                        .multilineTextAlignment(.center)
                }

                Button(action: authenticate) {
                    HStack(spacing: 8) {
                        Image(systemName: "faceid")
                            .font(.system(size: 16, weight: .semibold))
                        Text(isAuthenticating ? "Déverrouillage…" : "Déverrouiller MedLLM")
                            .font(.system(.body, design: .rounded, weight: .semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 28)
                    .padding(.vertical, 14)
                    .background(ChatTheme.accentGradient(load: 0), in: Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.white.opacity(0.22), lineWidth: 0.6)
                    )
                    .shadow(color: ChatTheme.electricBlue.opacity(0.38), radius: 16, y: 4)
                }
                .buttonStyle(.plain)
                .disabled(isAuthenticating)
            }
            .padding(32)
        }
    }
}

/// A visually rich but static background. Keeping it independent from the
/// observable chat model prevents token updates from rebuilding its layers.
struct ChatBackground: View {
    var body: some View {
        ZStack {
            MeshGradient(
                width: 3,
                height: 3,
                points: [
                    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                    [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                    [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
                ],
                colors: [
                    ChatTheme.navy, ChatTheme.indigo, ChatTheme.navy,
                    ChatTheme.deepViolet.opacity(0.52), ChatTheme.backgroundBase, ChatTheme.teal.opacity(0.42),
                    ChatTheme.indigo.opacity(0.70), ChatTheme.navy, ChatTheme.teal.opacity(0.30),
                ]
            )

            // Ambient glowing orbs for depth
            VStack {
                Circle()
                    .fill(ChatTheme.auroraCyan.opacity(0.08))
                    .frame(width: 280, height: 280)
                    .blur(radius: 60)
                    .offset(x: -80, y: -60)

                Spacer()

                Circle()
                    .fill(ChatTheme.deepViolet.opacity(0.12))
                    .frame(width: 320, height: 320)
                    .blur(radius: 80)
                    .offset(x: 90, y: 80)
            }

            LinearGradient(
                colors: [.clear, Color.black.opacity(0.15), Color.black.opacity(0.40)],
                startPoint: .top,
                endPoint: .bottom
            )
        }
        .drawingGroup(opaque: true, colorMode: .linear)
    }
}

#Preview {
    ContentView(viewModel: ChatViewModel(previewMode: true))
}
