import Foundation
import MLX
import MLXNN

/// Runner minimal pour exécuter un modèle MLX localement
@MainActor
final class MLXRunner {
    private var parameters: [String: MLXArray] = [:]
    
    init() {
        Task {
            await loadModel()
        }
    }
    
    /// Charge les poids sauvegardés depuis le bundle (phi3-medprof-merged)
    func loadModel() async {
        do {
            guard let modelURL = Bundle.main.url(forResource: "phi3-medprof-merged", withExtension: nil) else {
                print("❌ Impossible de trouver le dossier du modèle dans le bundle.")
                return
            }

            print("📦 Chargement du modèle MLX depuis :", modelURL.path)
            
            // Charger tous les fichiers poids (.npz, .safetensors, etc.)
            let fileManager = FileManager.default
            let files = try fileManager.contentsOfDirectory(atPath: modelURL.path)
            
            for file in files where file.hasSuffix(".npz") || file.hasSuffix(".safetensors") {
                let filePath = modelURL.appendingPathComponent(file).path
                print("🔹 Chargement des poids :", file)
                let arrays = try await MLX.load(filePath)
                for (k, v) in arrays {
                    parameters[k] = v
                }
            }
            
            print("✅ Modèle MLX chargé avec \(parameters.count) tenseurs.")
        } catch {
            print("❌ Erreur lors du chargement du modèle :", error.localizedDescription)
        }
    }
    
    /// Simule une génération de texte (temporairement)
    func generateResponse(for prompt: String) async -> String {
        guard !parameters.isEmpty else {
            return "⚠️ Modèle non chargé."
        }
        
        // ⚙️ Simulation temporaire
        print("🧠 Simulation d’inférence pour :", prompt)
        let response = "Le mécanisme physiologique de la sécrétion d’insuline implique la glycolyse du glucose dans les cellules β pancréatiques."
        return response
    }
}
