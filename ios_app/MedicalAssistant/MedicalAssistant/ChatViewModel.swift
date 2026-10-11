import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// ViewModel principal du chat médical
@MainActor
final class ChatViewModel: ObservableObject {
    // MARK: - Propriétés publiées (pour SwiftUI)
    @Published var sessions: [ChatSession] = []
    @Published var activeSessionId: UUID?
    @Published var messages: [Message] = []           // Historique des messages
    @Published var isGenerating: Bool = false
    @Published var errorMessage: String? = nil

    var activeSession: ChatSession? {
        sessions.first(where: { $0.id == activeSessionId })
    }

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
    private var memoryObserver: Any?

    // MARK: - Initialisation
    init(previewMode: Bool = false) {
        setupMemoryWarningObserver()

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
            Task { [weak self] in
                guard let self else { return }
                let loaded = await self.store.loadSessions()
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

    deinit {
        if let memoryObserver {
            NotificationCenter.default.removeObserver(memoryObserver)
        }
    }

    private func setupMemoryWarningObserver() {
        #if canImport(UIKit)
        memoryObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        #endif
    }

    private func handleMemoryWarning() {
        print("🚨 ChatViewModel: Avertissement de mémoire reçu d'iOS!")
        if isGenerating {
            Task { [weak self] in
                await self?._mlx?.clearInferenceCache()
            }
        } else {
            unloadModel()
        }
    }

    // MARK: - Envoi de message
    func sendMessage(_ text: String) {
        let inputText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !inputText.isEmpty, !isGenerating else { return }

        SpeechSynthesizerManager.shared.stop()

        // Ajoute le message utilisateur
        let userMessage = Message(text: inputText, isUser: true)
        messages.append(userMessage)
        let contextMessages = messages
        HapticsHelper.playLightImpact()
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
                    var lastRenderedResponse = ""
                    let renderClock = ContinuousClock()
                    var lastRender = renderClock.now
                    let stream = await runner.generateResponseStream(for: inputText, history: contextMessages)
                    for try await chunk in stream {
                        guard !Task.isCancelled else { return }
                        accumulatedResponse += chunk

                        // Coalesce very fast token streams. Re-parsing Markdown and laying out
                        // the entire chat for every token is one of the largest sources of UI hitches.
                        let now = renderClock.now
                        let shouldRender = lastRender.duration(to: now) >= .milliseconds(50)
                            || chunk.contains(where: { ".!?\n".contains($0) })
                        if shouldRender, accumulatedResponse != lastRenderedResponse {
                            self.updateMessage(id: assistantId, text: accumulatedResponse, isPartial: true)
                            HapticsHelper.playTokenTick(for: chunk)
                            lastRenderedResponse = accumulatedResponse
                            lastRender = now
                        }
                    }
                    guard !Task.isCancelled else { return }
                    self.updateMessage(id: assistantId, text: accumulatedResponse, isPartial: false)
                    self.isGenerating = false
                    self.persistMessages()
                    HapticsHelper.playLightImpact()
                    await runner.clearInferenceCache()
                } catch {
                    if !Task.isCancelled {
                        self.updateMessage(id: assistantId, text: "⚠️ Erreur: \(error.localizedDescription)", isPartial: false)
                    }
                    self.isGenerating = false
                    self.persistMessages()
                    await runner.clearInferenceCache()
                }
            } else {
                self.updateMessage(id: assistantId, text: "⚠️ Modèle MLX indisponible.", isPartial: false)
                self.isGenerating = false
                self.persistMessages()
            }
        }
    }

    func cancelGeneration() {
        SpeechSynthesizerManager.shared.stop()
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        if let idx = messages.lastIndex(where: { !$0.isUser && $0.isPartial }) {
            if messages[idx].text.isEmpty {
                messages[idx].text = "Génération interrompue."
            }
            messages[idx].isPartial = false
        }
        persistMessages()
        Task { [weak self] in
            await self?._mlx?.clearInferenceCache()
        }
    }

    // MARK: - Sessions Management
    
    func createNewSession() {
        SpeechSynthesizerManager.shared.stop()
        cancelGeneration()
        let newSession = ChatSession(title: "Nouvelle discussion", messages: [])
        sessions.insert(newSession, at: 0)
        activeSessionId = newSession.id
        messages = []
        persistMessages()
    }
    
    func switchSession(to id: UUID) {
        guard id != activeSessionId else { return }
        SpeechSynthesizerManager.shared.stop()
        cancelGeneration()
        if let session = sessions.first(where: { $0.id == id }) {
            activeSessionId = session.id
            messages = session.messages
        }
    }

    func renameSession(id: UUID, newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let idx = sessions.firstIndex(where: { $0.id == id }) {
            sessions[idx].title = trimmed
            sessions[idx].updatedAt = Date()
            persistMessages(delay: .zero)
        }
    }
    
    func deleteSession(id: UUID) {
        SpeechSynthesizerManager.shared.stop()
        sessions.removeAll(where: { $0.id == id })
        if activeSessionId == id {
            if let first = sessions.first {
                activeSessionId = first.id
                messages = first.messages
            } else {
                createNewSession()
            }
        }
        persistMessages(delay: .zero)
    }

    func clearAllSessions() {
        SpeechSynthesizerManager.shared.stop()
        cancelGeneration()
        sessions.removeAll()
        messages.removeAll()
        createNewSession()
        Task { [store] in
            await store.clearAll()
        }
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
        sendMessage(targetText)
    }

    func editMessage(id: UUID, newText: String) {
        guard !isGenerating else { return }
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        guard messages[idx].isUser else { return } // only edit user messages for simplicity
        
        // Remove everything after this message
        messages.removeSubrange(idx...)
        sendMessage(newText)
    }

    func unloadModel() {
        cancelGeneration()
        guard let runner = _mlx else { return }
        _mlx = nil
        Task(priority: .utility) {
            await runner.unloadModel()
        }
    }

    // MARK: - Helpers
    private func updateMessage(id: UUID, text: String, isPartial: Bool) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].text = text
        messages[idx].isPartial = isPartial
    }

    private func persistMessages(delay: Duration = .milliseconds(300)) {
        syncActiveSession()
        let snapshot = sessions
        saveTask?.cancel()
        saveTask = Task { [store] in
            if delay > .zero {
                try? await Task.sleep(for: delay)
            }
            guard !Task.isCancelled else { return }
            await store.saveSessions(snapshot)
        }
    }
}
