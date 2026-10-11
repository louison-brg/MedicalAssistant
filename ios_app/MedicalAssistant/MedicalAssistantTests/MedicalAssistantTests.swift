import Foundation
import Testing
@testable import MedicalAssistant

struct MedicalAssistantTests {

    @Test func testMessageInitializationAndProperties() {
        let id = UUID()
        let msg = Message(id: id, text: "J'ai de la fièvre", isUser: true)
        #expect(msg.id == id)
        #expect(msg.text == "J'ai de la fièvre")
        #expect(msg.isUser == true)
        #expect(msg.isPartial == false)
    }

    @Test func testChatSessionExportText() {
        let msg1 = Message(text: "Quels sont les effets secondaires du paracétamol ?", isUser: true)
        let msg2 = Message(text: "Le paracétamol est généralement bien toléré aux doses usuelles.", isUser: false)
        let session = ChatSession(title: "Conseils Paracétamol", messages: [msg1, msg2])

        let export = session.exportText()

        #expect(export.contains("Consultation MedLLM : Conseils Paracétamol"))
        #expect(export.contains("Question du patient"))
        #expect(export.contains("Quels sont les effets secondaires du paracétamol ?"))
        #expect(export.contains("Réponse MedLLM"))
        #expect(export.contains("Le paracétamol est généralement bien toléré aux doses usuelles."))
        #expect(export.contains("Ce document est généré à des fins d'information personnelle"))
    }

    @Test @MainActor func testChatViewModelSessionOperations() {
        let vm = ChatViewModel(previewMode: true)

        #expect(vm.sessions.count == 1)
        #expect(vm.messages.count == 2)
        #expect(vm.activeSessionId != nil)

        // Test rename
        let currentId = vm.activeSessionId!
        vm.renameSession(id: currentId, newTitle: "Céphalées matinales")
        #expect(vm.activeSession?.title == "Céphalées matinales")

        // Test create new session
        vm.createNewSession()
        #expect(vm.sessions.count == 2)
        #expect(vm.messages.isEmpty)
        #expect(vm.activeSession?.title == "Nouvelle discussion")

        // Test switch session
        vm.switchSession(to: currentId)
        #expect(vm.activeSessionId == currentId)
        #expect(vm.messages.count == 2)

        // Test delete session
        let newId = vm.sessions.first(where: { $0.id != currentId })!.id
        vm.deleteSession(id: newId)
        #expect(vm.sessions.count == 1)
        #expect(vm.sessions.first?.id == currentId)
    }

    @Test @MainActor func testChatViewModelMessageEditing() {
        let vm = ChatViewModel(previewMode: true)
        #expect(vm.messages.count == 2)

        let firstMsg = vm.messages[0]
        #expect(firstMsg.isUser == true)

        // Editing removes everything from that point onwards
        vm.editMessage(id: firstMsg.id, newText: "Nouveau message édité")
        // Since MLX model is not loaded in preview mode, it handles message replacement
        #expect(vm.messages.count >= 1)
        #expect(vm.messages[0].text == "Nouveau message édité")
    }

    @Test @MainActor func testChatViewModelClearMessages() {
        let vm = ChatViewModel(previewMode: true)
        #expect(!vm.messages.isEmpty)

        vm.clearMessages()
        #expect(vm.messages.isEmpty)
    }

    @Test func testMessageStoreSaveAndLoad() async {
        let store = MessageStore()
        let testMsg = Message(text: "Test persistance", isUser: true)
        let testSession = ChatSession(id: UUID(), title: "Test Session", messages: [testMsg])

        await store.saveSessions([testSession])
        let loaded = await store.loadSessions()

        #expect(loaded.contains(where: { $0.id == testSession.id }))
        let matched = loaded.first(where: { $0.id == testSession.id })
        #expect(matched?.title == "Test Session")
        #expect(matched?.messages.first?.text == "Test persistance")
    }

    @Test func testTokenizerBasicPropertiesAndCacheClearing() {
        let tokenizer = Tokenizer()
        #expect(tokenizer.eosTokenId >= 0)
        #expect(tokenizer.bosTokenId >= 0)
        #expect(tokenizer.padTokenId >= 0)

        // Verify cache clearing executes safely
        tokenizer.clearCache()
        #expect(tokenizer.isValid(id: tokenizer.eosTokenId) || tokenizer.eosTokenId == 32000)
    }

    @Test @MainActor func testSpeechSynthesizerManagerInitialState() {
        let manager = SpeechSynthesizerManager.shared
        #expect(manager.speakingMessageId == nil)
        #expect(manager.isSpeaking(messageId: UUID()) == false)
        manager.stop()
        #expect(manager.speakingMessageId == nil)
    }
}
