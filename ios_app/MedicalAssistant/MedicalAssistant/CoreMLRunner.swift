import Foundation
import CoreML

/// 🧠 Classe utilitaire qui encapsule le modèle CoreML Phi-3
final class CoreMLRunner {
    private var model: MLModel?
    private let tokenizer = Tokenizer()

    init() {
        loadModel()
    }

    /// Charge le modèle CoreML (.mlpackage compilé en .mlmodelc)
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "MedicalLLM", withExtension: "mlmodelc") else {
            print("❌ Impossible de trouver MedicalLLM.mlmodelc dans le bundle.")
            return
        }

        do {
            let config = MLModelConfiguration()
            model = try MLModel(contentsOf: modelURL, configuration: config)
            print("✅ Modèle CoreML chargé avec succès :", modelURL.lastPathComponent)
        } catch {
            print("⚠️ Erreur de chargement du modèle :", error.localizedDescription)
        }
    }

    /// Effectue une inférence sur du texte d’entrée
    func generateResponse(for text: String) -> String {
        guard let model = model else {
            return "⚠️ Modèle non chargé."
        }

        // 1️⃣ Tokenisation
        let tokens = tokenizer.encode(text)
        guard !tokens.isEmpty else {
            return "⚠️ Aucun token généré pour cette entrée."
        }

        do {
            // 2️⃣ Création de l’entrée CoreML
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: tokens.count)], dataType: .int32)
            for (i, token) in tokens.enumerated() {
                inputArray[i] = NSNumber(value: token)
            }

            let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])

            // 3️⃣ Exécution
            let output = try model.prediction(from: input)

            // 4️⃣ Récupération des tokens générés
            let outputKey = output.featureNames.first ?? ""
            guard let resultArray = output.featureValue(for: outputKey)?.multiArrayValue else {
                return "⚠️ Aucune sortie trouvée."
            }

            var resultTokens: [Int] = []
            for i in 0..<resultArray.count {
                resultTokens.append(Int(truncating: resultArray[i]))
            }

            // 5️⃣ Décodage
            let decoded = tokenizer.decode(resultTokens)
            return decoded.isEmpty ? "(Réponse vide)" : decoded

        } catch {
            print("❌ Erreur d’inférence :", error.localizedDescription)
            return "⚠️ Erreur pendant la génération."
        }
    }
}
