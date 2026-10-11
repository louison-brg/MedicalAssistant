import Foundation

//  Message.swift
//  MedicalAssistant
//
//  Created by Louison Beranger on 17/10/2025.
struct Message: Identifiable, Codable, Equatable {
    let id: UUID
    var text: String
    let isUser: Bool // true = utilisateur, false = modele
    let createdAt: Date
    var isPartial: Bool

    init(id: UUID = UUID(), text: String, isUser: Bool, createdAt: Date = Date(), isPartial: Bool = false) {
        self.id = id
        self.text = text
        self.isUser = isUser
        self.createdAt = createdAt
        self.isPartial = isPartial
    }
}

struct ChatSession: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    var messages: [Message]
    let createdAt: Date
    var updatedAt: Date
    
    init(id: UUID = UUID(), title: String, messages: [Message] = [], createdAt: Date = Date(), updatedAt: Date = Date()) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// Generates clean Markdown export suitable for sharing with a doctor or personal records.
    func exportText() -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short

        var export = "# Consultation MedLLM : \(title)\n"
        export += "Date : \(formatter.string(from: updatedAt))\n"
        export += "Messages : \(messages.count)\n\n"
        export += "---\n\n"

        for msg in messages {
            let role = msg.isUser ? "👤 Question du patient" : "🩺 Réponse MedLLM"
            let timeStr = formatter.string(from: msg.createdAt)
            export += "### \(role) (\(timeStr))\n\n"
            export += "\(msg.text.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
        }

        export += "---\n"
        export += "*Note : Ce document est généré à des fins d'information personnelle via un modèle d'IA local. Il ne remplace en aucun cas une consultation médicale, un diagnostic ou une prescription formelle.*\n"
        return export
    }
}
