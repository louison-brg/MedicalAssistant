import Foundation

/// 🔤 Tokenizer simplifié compatible avec ton modèle Phi-3 fine-tuné (BPE-style)
final class Tokenizer {
    private var vocab: [String: Int] = [:]
    private var merges: [(String, String)] = []
    private var idToToken: [Int: String] = [:]

    init() {
        loadFiles()
    }

    /// Charge `vocab.json` et `merges.txt` depuis le bundle
    private func loadFiles() {
        guard
            let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json"),
            let mergesURL = Bundle.main.url(forResource: "merges", withExtension: "txt")
        else {
            print("❌ Impossible de trouver vocab.json ou merges.txt dans le bundle.")
            return
        }

        do {
            // Charger le vocabulaire JSON
            let vocabData = try Data(contentsOf: vocabURL)
            if let dict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
                vocab = dict
                idToToken = Dictionary(uniqueKeysWithValues: dict.map { ($1, $0) })
            }

            // Charger les merges (les paires BPE)
            let mergesText = try String(contentsOf: mergesURL)
            merges = mergesText
                .components(separatedBy: .newlines)
                .filter { !$0.hasPrefix("#") && !$0.isEmpty }
                .map {
                    let parts = $0.split(separator: " ")
                    return (String(parts[0]), String(parts[1]))
                }

            print("✅ Tokenizer Phi-3 chargé avec \(vocab.count) tokens.")
        } catch {
            print("❌ Erreur lors du chargement du tokenizer :", error.localizedDescription)
        }
    }

    // MARK: - Encodage

    /// Encode une phrase en IDs de tokens
    func encode(_ text: String) -> [Int] {
        // Version simplifiée : découpe sur les espaces (phi-3 utilise BPE en vrai)
        return text
            .split(separator: " ")
            .compactMap { vocab[String($0)] ?? vocab["<unk>"] }
    }

    // MARK: - Décodage

    /// Convertit des IDs en texte lisible
    func decode(_ tokens: [Int]) -> String {
        return tokens.compactMap { idToToken[$0] }.joined(separator: " ")
    }

    /// ID du token de fin (EOS)
    var eosTokenId: Int {
        vocab["<eos>"] ?? vocab["</s>"] ?? vocab["<end>"] ?? -1
    }
}
