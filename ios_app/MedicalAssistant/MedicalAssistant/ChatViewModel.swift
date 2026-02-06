import Foundation

/// ViewModel principal du chat médical
@MainActor
final class ChatViewModel: ObservableObject {
    // MARK: - Propriétés publiées (pour SwiftUI)
    @Published var messages: [Message] = []           // Historique des messages
    @Published var currentInput: String = ""          // Texte saisi par l’utilisateur
    @Published var isGenerating: Bool = false
    @Published var errorMessage: String? = nil

    // MARK: - Composants internes
    private lazy var coreML: CoreMLRunner? = {
        print("⚙️ Chargement paresseux du modèle CoreML…")
        return CoreMLRunner(preferGPU: true)
    }()

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
            print("🧩 ChatViewModel lancé en mode Preview — modèle non chargé.")
        } else {
            self.messages = store.load()
            print("🧠 ChatViewModel prêt à utiliser le modèle CoreML.")
        }
    }

    // MARK: - Envoi de message
    func sendMessage() {
        let inputText = currentInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !inputText.isEmpty, !isGenerating else { return }

        // Ajoute le message utilisateur
        let userMessage = Message(text: inputText, isUser: true)
        messages.append(userMessage)
        HapticsHelper.playLightImpact()
        currentInput = ""
        errorMessage = nil
        persistMessages()

        guard let runner = coreML else {
            print("⚠️ Modèle CoreML indisponible.")
            errorMessage = "Model not available. Please check the CoreML bundle."
            let fallback = Message(text: "Model not available right now. Please try again shortly.", isUser: false)
            messages.append(fallback)
            persistMessages()
            return
        }

        let assistantMessage = Message(text: "", isUser: false, isPartial: true)
        let assistantId = assistantMessage.id
        messages.append(assistantMessage)
        persistMessages()

        isGenerating = true
        generationTask?.cancel()
        generationTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            let response = runner.generateResponse(
                for: inputText,
                onPartial: { partial in
                    Task { @MainActor in
                        self.updateMessage(id: assistantId, text: partial, isPartial: true)
                    }
                },
                shouldCancel: { Task.isCancelled }
            )
            await MainActor.run {
                self.updateMessage(id: assistantId, text: response, isPartial: false)
                self.isGenerating = false
                self.persistMessages()
                HapticsHelper.playLightImpact()
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

    func clearMessages() {
        cancelGeneration()
        messages.removeAll()
        store.clear()
    }

    func setPerformance(preferGPU: Bool) {
        if let runner = coreML {
            runner.setPerformance(preferGPU: preferGPU)
        } else {
            coreML = CoreMLRunner(preferGPU: preferGPU)
        }
    }

    // MARK: - Helpers
    private func updateMessage(id: UUID, text: String, isPartial: Bool) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].text = text
        messages[idx].isPartial = isPartial
    }

    private func persistMessages() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 300_000_000)
            await MainActor.run {
                guard let self else { return }
                self.store.save(self.messages)
            }
        }
    }
}
