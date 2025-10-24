import Foundation
import Combine

/// ViewModel principal du chat médical
final class ChatViewModel: ObservableObject {
    // MARK: - Propriétés publiées (pour SwiftUI)
    @Published var messages: [Message] = []           // Historique des messages
    @Published var currentInput: String = ""          // Texte saisi par l’utilisateur

    // MARK: - Composants internes
    private let mlx = MLXRunner()
           // Moteur d’inférence CoreML
    private var cancellables = Set<AnyCancellable>()  // Pour Combine

    // MARK: - Initialisation
    init(previewMode: Bool = false) {
        if previewMode {
            // État de démo pour les Previews Xcode
            self.messages = [
                Message(text: "Bonjour docteur, j’ai mal à la tête depuis ce matin.", isUser: true),
                Message(text: "Avez-vous pris votre température ? Cela pourrait être une simple infection virale.", isUser: false)
            ]
            print("🧩 ChatViewModel lancé en mode Preview — modèle non chargé.")
        } else {
            print("🧠 ChatViewModel prêt à utiliser le modèle CoreML.")
        }
    }

    // MARK: - Envoi de message
    func sendMessage() {
        let inputText = currentInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !inputText.isEmpty else { return }

        // Ajoute le message utilisateur
        let userMessage = Message(text: inputText, isUser: true)
        messages.append(userMessage)
        currentInput = ""

        // Lance la génération sur un thread secondaire
        DispatchQueue.global(qos: .userInitiated).async {
            Task {
                let botReply = await self.mlx.generateResponse(for: inputText)
                await MainActor.run {
                    self.messages.append(Message(text: botReply, isUser: false))
                }
            }
        }

    }
}
