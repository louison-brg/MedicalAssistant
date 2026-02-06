import Foundation
#if canImport(CoreML)
import CoreML
#endif

/// Utilitaires pour charger un modèle Core ML avec une configuration sûre
/// afin d'éviter les erreurs MPSGraph/Expresso sur OS non compatibles.
enum CoreMLConfig {
    #if canImport(CoreML)
    /// Retourne une configuration par défaut qui privilégie la compatibilité.
    /// - Sur machines non compatibles GPU/MPSGraph, bascule en `.cpuOnly`.
    static func safeConfiguration(preferGPU: Bool = true) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if preferGPU {
            // Essaye GPU si dispo, sinon CPU
            config.computeUnits = .cpuAndGPU
        } else {
            config.computeUnits = .cpuOnly
        }
        return config
    }

    /// Charge un modèle à partir d'une URL avec la configuration sûre.
    static func loadModel(at url: URL, preferGPU: Bool = true) throws -> MLModel {
        let config = safeConfiguration(preferGPU: preferGPU)
        do {
            return try MLModel(contentsOf: url, configuration: config)
        } catch {
            // Fallback CPU-only si l'ouverture échoue (souvent à cause de MPSGraph)
            let cpuConfig = safeConfiguration(preferGPU: false)
            return try MLModel(contentsOf: url, configuration: cpuConfig)
        }
    }

    /// Exemple de helper pour charger un modèle généré `MedicalLLM` si présent.
    /// Cette fonction n'échoue pas à la compilation si le type n'existe pas.
    static func loadMedicalLLM(preferGPU: Bool = true) -> Any? {
        // Utilise réflexion pour éviter la dépendance directe.
        guard let modelType = NSClassFromString("MedicalLLM") as? NSObject.Type else {
            return nil
        }
        let selector = NSSelectorFromString("init(configuration:)")
        let config = safeConfiguration(preferGPU: preferGPU)
        if modelType.instancesRespond(to: selector) {
            // Tente initialisation avec config, sinon fallback CPU
            if let instance = modelType.alloc() as? NSObject {
                let _ = instance.perform(selector, with: config)
                return instance
            }
        }
        // Fallback CPU only
        let cpuConfig = safeConfiguration(preferGPU: false)
        if modelType.instancesRespond(to: selector) {
            if let instance = modelType.alloc() as? NSObject {
                let _ = instance.perform(selector, with: cpuConfig)
                return instance
            }
        }
        return nil
    }
    #endif
}
