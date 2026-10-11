import SwiftUI

struct ChatHistoryView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var searchText = ""
    @State private var sessionToRename: ChatSession?
    @State private var renameText = ""
    @State private var isShowingRenameAlert = false
    @State private var isShowingClearAllAlert = false

    private var filteredSessions: [ChatSession] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if query.isEmpty {
            return viewModel.sessions
        }
        return viewModel.sessions.filter { session in
            session.title.localizedCaseInsensitiveContains(query) ||
            session.messages.contains { $0.text.localizedCaseInsensitiveContains(query) }
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                ChatTheme.backgroundBase.ignoresSafeArea()

                if viewModel.sessions.isEmpty {
                    emptyState
                } else if filteredSessions.isEmpty {
                    emptySearchState
                } else {
                    sessionList
                }
            }
            .navigationTitle("Conversations")
            .navigationBarTitleDisplayMode(.inline)
            .searchable(text: $searchText, prompt: "Rechercher une consultation…")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Fermer") {
                        dismiss()
                    }
                    .font(.system(.body, design: .rounded, weight: .medium))
                    .foregroundStyle(.white.opacity(0.85))
                }

                ToolbarItemGroup(placement: .primaryAction) {
                    if !viewModel.sessions.isEmpty {
                        Menu {
                            Button(role: .destructive) {
                                isShowingClearAllAlert = true
                            } label: {
                                Label("Tout effacer", systemImage: "trash")
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                                .font(.system(size: 16))
                                .foregroundStyle(.white.opacity(0.8))
                        }
                    }

                    Button {
                        HapticsHelper.playLightImpact()
                        viewModel.createNewSession()
                        dismiss()
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "square.and.pencil")
                                .font(.system(size: 13, weight: .bold))
                            Text("Nouveau")
                                .font(.system(.subheadline, design: .rounded, weight: .semibold))
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(ChatTheme.userBubbleGradient, in: Capsule())
                    }
                    .accessibilityLabel("Nouvelle conversation")
                }
            }
            .alert("Renommer la discussion", isPresented: $isShowingRenameAlert) {
                TextField("Titre de la consultation", text: $renameText)
                Button("Annuler", role: .cancel) {}
                Button("Enregistrer") {
                    if let session = sessionToRename {
                        viewModel.renameSession(id: session.id, newTitle: renameText)
                    }
                }
            }
            .confirmationDialog(
                "Supprimer toutes les conversations ?",
                isPresented: $isShowingClearAllAlert,
                titleVisibility: .visible
            ) {
                Button("Supprimer tout l'historique", role: .destructive) {
                    HapticsHelper.playLightImpact()
                    viewModel.clearAllSessions()
                    dismiss()
                }
                Button("Annuler", role: .cancel) {}
            } message: {
                Text("Cette action supprimera définitivement toutes vos consultations enregistrées sur cet appareil.")
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(ChatTheme.electricBlue.opacity(0.12))
                    .frame(width: 80, height: 80)
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(ChatTheme.auroraCyan)
            }
            .shadow(color: ChatTheme.electricBlue.opacity(0.2), radius: 16)

            Text("Aucune conversation")
                .font(.system(.title3, design: .rounded, weight: .bold))
                .foregroundStyle(.white)

            Text("Démarrez une consultation médicale pour retrouver ici vos questions et réponses privées.")
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white.opacity(0.5))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }

    private var emptySearchState: some View {
        VStack(spacing: 14) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 36))
                .foregroundStyle(.white.opacity(0.3))

            Text("Aucun résultat")
                .font(.system(.headline, design: .rounded, weight: .bold))
                .foregroundStyle(.white.opacity(0.85))

            Text("Aucune consultation ne correspond à « \(searchText) ».")
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white.opacity(0.5))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
    }

    private var sessionList: some View {
        List {
            ForEach(filteredSessions) { session in
                sessionCard(session)
                    .listRowInsets(EdgeInsets(top: 6, leading: 16, bottom: 6, trailing: 16))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(role: .destructive) {
                            withAnimation {
                                viewModel.deleteSession(id: session.id)
                            }
                        } label: {
                            Label("Supprimer", systemImage: "trash")
                        }
                    }
                    .swipeActions(edge: .leading) {
                        Button {
                            beginRenaming(session)
                        } label: {
                            Label("Renommer", systemImage: "pencil")
                        }
                        .tint(ChatTheme.electricBlue)
                    }
                    .contextMenu {
                        Button {
                            beginRenaming(session)
                        } label: {
                            Label("Renommer", systemImage: "pencil")
                        }

                        ShareLink(item: session.exportText()) {
                            Label("Partager la consultation", systemImage: "square.and.arrow.up")
                        }

                        Divider()

                        Button(role: .destructive) {
                            withAnimation {
                                viewModel.deleteSession(id: session.id)
                            }
                        } label: {
                            Label("Supprimer", systemImage: "trash")
                        }
                    }
            }
            .onDelete(perform: deleteSessions)
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
    }

    private func sessionCard(_ session: ChatSession) -> some View {
        let isActive = session.id == viewModel.activeSessionId
        let snippet = session.messages.last?.text ?? session.messages.first?.text

        return Button {
            HapticsHelper.playLightImpact()
            viewModel.switchSession(to: session.id)
            dismiss()
        } label: {
            HStack(spacing: 14) {
                // Icon
                ZStack {
                    Circle()
                        .fill(
                            isActive
                                ? AnyShapeStyle(ChatTheme.userBubbleGradient)
                                : AnyShapeStyle(Color.white.opacity(0.06))
                        )
                    Image(systemName: isActive ? "stethoscope" : "bubble.left.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(isActive ? .white : .white.opacity(0.5))
                }
                .frame(width: 40, height: 40)
                .shadow(
                    color: isActive ? ChatTheme.royalBlue.opacity(0.4) : .clear,
                    radius: 6,
                    y: 2
                )

                // Details
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(session.title)
                            .font(.system(.body, design: .rounded, weight: .semibold))
                            .foregroundStyle(isActive ? .white : .white.opacity(0.92))
                            .lineLimit(1)

                        Spacer()

                        if isActive {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(ChatTheme.mint)
                                    .frame(width: 5, height: 5)
                                Text("Active")
                                    .font(.system(size: 10, weight: .bold, design: .rounded))
                            }
                            .foregroundStyle(ChatTheme.mint)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(ChatTheme.mint.opacity(0.12), in: Capsule())
                        }
                    }

                    if let snippet, !snippet.isEmpty {
                        Text(snippet.trimmingCharacters(in: .whitespacesAndNewlines))
                            .font(.system(size: 12, design: .rounded))
                            .foregroundStyle(.white.opacity(0.45))
                            .lineLimit(1)
                    }

                    HStack(spacing: 6) {
                        Text("\(session.messages.count) messages")
                        Text("·")
                        Text(session.updatedAt, style: .relative)
                    }
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.32))
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(isActive ? Color.white.opacity(0.08) : Color.white.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(
                        isActive
                            ? ChatTheme.auroraCyan.opacity(0.4)
                            : Color.white.opacity(0.08),
                        lineWidth: isActive ? 0.9 : 0.5
                    )
            )
            .shadow(
                color: isActive ? ChatTheme.electricBlue.opacity(0.18) : Color.black.opacity(0.12),
                radius: 8,
                y: 2
            )
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private func beginRenaming(_ session: ChatSession) {
        sessionToRename = session
        renameText = session.title
        isShowingRenameAlert = true
    }

    private func deleteSessions(at offsets: IndexSet) {
        let ids = offsets.map { filteredSessions[$0].id }
        ids.forEach(viewModel.deleteSession)
    }
}
