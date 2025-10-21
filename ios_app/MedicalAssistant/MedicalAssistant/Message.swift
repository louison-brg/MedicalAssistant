import Foundation

//  Message.swift
//  MedicalAssistant
//
//  Created by Louison Beranger on 17/10/2025.
struct Message: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool // true = utilisateur, false = modele
}
