import Foundation

@MainActor
final class MessageStore {
    private let fileName = "sessions.json"
    private let legacyFileName = "messages.json"

    func loadSessions() -> [ChatSession] {
        // 🔥 Migration Step: Migrate old single chat into a ChatSession
        if let legacyUrl = fileURL(for: legacyFileName), FileManager.default.fileExists(atPath: legacyUrl.path) {
            do {
                let data = try Data(contentsOf: legacyUrl)
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let oldMessages = try decoder.decode([Message].self, from: data)
                
                let migratedSession = ChatSession(
                    title: "Ancienne discussion",
                    messages: oldMessages,
                    createdAt: oldMessages.first?.createdAt ?? Date(),
                    updatedAt: oldMessages.last?.createdAt ?? Date()
                )
                
                // Save it to the new format right away
                saveSessions([migratedSession])
                
                // Remove legacy file
                try? FileManager.default.removeItem(at: legacyUrl)
                print("🔄 Migrated legacy messages to new ChatSession format.")
                
                return [migratedSession]
            } catch {
                print("⚠️ Failed to migrate legacy messages: \(error.localizedDescription)")
            }
        }
        
        guard let url = fileURL(for: fileName), FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([ChatSession].self, from: data)
        } catch {
            print("⚠️ Failed to load sessions: \(error.localizedDescription)")
            return []
        }
    }

    func saveSessions(_ sessions: [ChatSession]) {
        guard let url = fileURL(for: fileName) else { return }
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted]
            let data = try encoder.encode(sessions)
            try data.write(to: url, options: [.atomic, .completeFileProtection])
        } catch {
            print("⚠️ Failed to save sessions: \(error.localizedDescription)")
        }
    }

    func clearAll() {
        if let url = fileURL(for: fileName) { try? FileManager.default.removeItem(at: url) }
        if let url = fileURL(for: legacyFileName) { try? FileManager.default.removeItem(at: url) }
    }

    private func fileURL(for name: String) -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(name)
    }
}
