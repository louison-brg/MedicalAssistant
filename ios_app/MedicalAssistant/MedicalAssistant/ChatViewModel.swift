import Foundation

/// ViewModel principal du chat médical
@MainActor
final class ChatViewModel: ObservableObject {
    // MARK: - Propriétés publiées (pour SwiftUI)
    @Published var sessions: [ChatSession] = []
    @Published var activeSessionId: UUID?
    @Published var messages: [Message] = []           // Historique des messages
    @Published var currentInput: String = ""          // Texte saisi par l’utilisateur
    @Published var isGenerating: Bool = false
    @Published var errorMessage: String? = nil
    @Published private(set) var lastSubmittedUserMessageID: UUID?

    // MARK: - Composants internes
    private var _mlx: MLXRunner?
    private var mlx: MLXRunner? {
        get {
            if _mlx == nil {
                print("⚙️ Chargement paresseux du modèle MLX…")
                _mlx = MLXRunner()
            }
            return _mlx
        }
        set {
            _mlx = newValue
        }
    }

    private let store = MessageStore()
    private var saveTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?

    // MARK: - Initialisation
    init(previewMode: Bool = false) {
        if previewMode {
            self.messages = [
                Message(text: "Bonjour docteur, j’ai mal à la tête depuis ce matin.", isUser: true),
                Message(text: "Avez-vous pris votre température ? Cela pourrait être une simple infection virale.", isUser: false)
            ]
            let previewSession = ChatSession(id: UUID(), title: "Preview", messages: self.messages)
            self.sessions = [previewSession]
            self.activeSessionId = previewSession.id
            print("🧩 ChatViewModel lancé en mode Preview — modèle non chargé.")
        } else {
            // We use Task because loadSessions could theoretically block, but here it's swift
            Task {
                let loaded = self.store.loadSessions()
                self.sessions = loaded
                if let recent = loaded.sorted(by: { $0.updatedAt > $1.updatedAt }).first {
                    self.activeSessionId = recent.id
                    self.messages = recent.messages
                } else {
                    self.createNewSession()
                }
            }
            print("🧠 ChatViewModel prêt à utiliser MLX.")
        }
    }

    // MARK: - Envoi de message
    func sendMessage() {
        let inputText = currentInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !inputText.isEmpty, !isGenerating else { return }

        // Ajoute le message utilisateur
        let userMessage = Message(text: inputText, isUser: true)
        messages.append(userMessage)
        lastSubmittedUserMessageID = userMessage.id
        let contextMessages = messages
        HapticsHelper.playLightImpact()
        currentInput = ""
        errorMessage = nil
        persistMessages()

        let assistantMessage = Message(text: "", isUser: false, isPartial: true)
        let assistantId = assistantMessage.id
        messages.append(assistantMessage)
        persistMessages()

        isGenerating = true
        generationTask?.cancel()
        generationTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            
            if let runner = self.mlx {
                do {
                    var accumulatedResponse = ""
                    let stream = await runner.generateResponseStream(for: inputText, history: contextMessages)
                    for try await chunk in stream {
                        accumulatedResponse += chunk
                        if !chunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            await MainActor.run {
                                HapticsHelper.playTokenTick(for: chunk)
                            }
                        }
                        await MainActor.run {
                            self.updateMessage(id: assistantId, text: accumulatedResponse, isPartial: true)
                        }
                    }
                    await MainActor.run {
                        self.updateMessage(id: assistantId, text: accumulatedResponse, isPartial: false)
                        self.isGenerating = false
                        self.persistMessages()
                        HapticsHelper.playLightImpact()
                    }
                } catch {
                    await MainActor.run {
                        if !Task.isCancelled {
                            self.updateMessage(id: assistantId, text: "⚠️ Erreur: \(error.localizedDescription)", isPartial: false)
                        }
                        self.isGenerating = false
                        self.persistMessages()
                    }
                }
            } else {
                await MainActor.run {
                    self.updateMessage(id: assistantId, text: "⚠️ Modèle MLX indisponible.", isPartial: false)
                    self.isGenerating = false
                    self.persistMessages()
                }
            }
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        if let idx = messages.lastIndex(where: { !$0.isUser && $0.isPartial }) {
            if messages[idx].text.isEmpty {
                messages[idx].text = "(Generation cancelled)"
            }
            messages[idx].isPartial = false
        }
        persistMessages()
    }

    // MARK: - Sessions Management
    
    func createNewSession() {
        cancelGeneration()
        let newSession = ChatSession(title: "Nouvelle discussion", messages: [])
        sessions.insert(newSession, at: 0)
        activeSessionId = newSession.id
        messages = []
        persistMessages()
    }
    
    func switchSession(to id: UUID) {
        guard id != activeSessionId else { return }
        cancelGeneration()
        if let session = sessions.first(where: { $0.id == id }) {
            activeSessionId = session.id
            messages = session.messages
        }
    }
    
    func deleteSession(id: UUID) {
        sessions.removeAll(where: { $0.id == id })
        if activeSessionId == id {
            if let first = sessions.first {
                activeSessionId = first.id
                messages = first.messages
            } else {
                createNewSession()
            }
        }
        store.saveSessions(sessions)
    }

    private func syncActiveSession() {
        guard let id = activeSessionId, let idx = sessions.firstIndex(where: { $0.id == id }) else { return }
        
        sessions[idx].messages = messages
        sessions[idx].updatedAt = Date()
        
        if sessions[idx].title == "Nouvelle discussion", let firstUserMsg = messages.first(where: { $0.isUser }) {
            let words = firstUserMsg.text.split(separator: " ")
            let titleStr = words.prefix(4).joined(separator: " ")
            sessions[idx].title = titleStr + (words.count > 4 ? "..." : "")
        }
        
        sessions.sort(by: { $0.updatedAt > $1.updatedAt })
    }

    func clearMessages() {
        cancelGeneration()
        messages.removeAll()
        persistMessages()
    }

    func regenerate(from id: UUID) {
        guard !isGenerating else { return }
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        
        // Find the last user message before or at this point
        var targetUserIdx = idx
        if !messages[idx].isUser {
            targetUserIdx = idx - 1
            guard targetUserIdx >= 0, messages[targetUserIdx].isUser else { return }
        }
        
        let targetText = messages[targetUserIdx].text
        
        // Remove everything after the target user message
        messages.removeSubrange(targetUserIdx...)
        currentInput = targetText
        sendMessage()
    }

    func editMessage(id: UUID, newText: String) {
        guard !isGenerating else { return }
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        guard messages[idx].isUser else { return } // only edit user messages for simplicity
        
        // Remove everything after this message
        messages.removeSubrange(idx...)
        currentInput = newText
        sendMessage()
    }

    func setPerformance(preferGPU: Bool) {
        _ = preferGPU
        print("ℹ️ Le toggle CPU/GPU est maintenu pour l'UI, mais on ne recharge pas le modèle pour éviter les fuites.")
    }

    func unloadModel() {
        print("🧹 Mise en arrière-plan. On maintient le modèle en mémoire : iOS compressera la RAM automatiquement.")
        cancelGeneration()
    }

    func clearComposerMorph() {
        lastSubmittedUserMessageID = nil
    }

    // MARK: - Helpers
    private func updateMessage(id: UUID, text: String, isPartial: Bool) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].text = text
        messages[idx].isPartial = isPartial
    }

    private func persistMessages() {
        syncActiveSession()
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 300_000_000)
            await MainActor.run {
                guard let self = self else { return }
                self.store.saveSessions(self.sessions)
            }
        }
    }
}
