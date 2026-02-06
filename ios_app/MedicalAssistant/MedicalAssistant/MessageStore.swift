import Foundation

@MainActor
final class MessageStore {
    private let fileName = "messages.json"

    func load() -> [Message] {
        guard let url = fileURL(), FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([Message].self, from: data)
        } catch {
            print("⚠️ Failed to load messages:", error.localizedDescription)
            return []
        }
    }

    func save(_ messages: [Message]) {
        guard let url = fileURL() else { return }
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted]
            let data = try encoder.encode(messages)
            try data.write(to: url, options: [.atomic])
        } catch {
            print("⚠️ Failed to save messages:", error.localizedDescription)
        }
    }

    func clear() {
        guard let url = fileURL(), FileManager.default.fileExists(atPath: url.path) else { return }
        try? FileManager.default.removeItem(at: url)
    }

    private func fileURL() -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(fileName)
    }
}
