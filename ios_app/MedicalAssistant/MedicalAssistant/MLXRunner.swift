import Foundation
import MLX
import MLXNN

#if canImport(MLXData)
import MLXData
#endif

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
                let fileURL = modelURL.appendingPathComponent(file)
                print("🔹 Chargement des poids :", file)
#if canImport(MLXData)
                if let arrays = try await loadArrays(from: fileURL) {
                    for (k, v) in arrays {
                        parameters[k] = v
                    }
                }
#else
                print("⚠️ Aucun chargeur MLX disponible pour \(file). Ajoutez le module MLXData ou implémentez un loader.")
#endif
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
    
    private func loadArrays(from url: URL) async throws -> [String: MLXArray]? {
#if canImport(MLXData)
        // Détecte le format en fonction de l'extension et délègue au loader MLXData si disponible.
        let path = url.path.lowercased()
        if path.hasSuffix(".npz") {
            // Exemple indicatif: adaptez le nom de type/fonction selon MLXData 0.29.3
            if let npzLoader = NSClassFromString("MLXData.NPZLoader") as? NSObject.Type,
               npzLoader.responds(to: Selector(("loadTensorsAtURL:error:"))) {
                // Appel dynamique laissé à titre de compatibilité; si votre API est différente, remplacez par l'appel direct.
                print("ℹ️ Chargement NPZ via MLXData pour: \(url.lastPathComponent)")
            }
            // TODO: Remplacer par l'appel direct MLXData une fois la signature confirmée.
            return [:]
        } else if path.hasSuffix(".safetensors") {
            if let stLoader = NSClassFromString("MLXData.SafeTensorsLoader") as? NSObject.Type,
               stLoader.responds(to: Selector(("loadTensorsAtURL:error:"))) {
                print("ℹ️ Chargement SafeTensors via MLXData pour: \(url.lastPathComponent)")
            }
            // TODO: Remplacer par l'appel direct MLXData une fois la signature confirmée.
            return [:]
        } else {
            print("⚠️ Format non supporté: \(url.lastPathComponent)")
            return nil
        }
#else
        return nil
#endif
    }
}

