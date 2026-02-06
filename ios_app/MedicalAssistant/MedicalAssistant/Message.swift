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
