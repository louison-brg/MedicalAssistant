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
    static func safeConfiguration(preferGPU: Bool = true, computeUnits: MLComputeUnits? = nil) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if let units = computeUnits {
            config.computeUnits = units
        } else if preferGPU {
            // Essaye GPU si dispo, sinon CPU
            config.computeUnits = .cpuAndGPU
        } else {
            config.computeUnits = .cpuOnly
        }
        return config
    }

    /// Charge un modèle à partir d'une URL avec la configuration sûre.
    static func loadModel(at url: URL, preferGPU: Bool = true, computeUnits: MLComputeUnits? = nil) throws -> MLModel {
        let config = safeConfiguration(preferGPU: preferGPU, computeUnits: computeUnits)
        do {
            return try MLModel(contentsOf: url, configuration: config)
        } catch {
            // Fallback CPU-only si l'ouverture échoue (souvent à cause de MPSGraph)
            let cpuConfig = safeConfiguration(preferGPU: false, computeUnits: computeUnits ?? .cpuOnly)
            return try MLModel(contentsOf: url, configuration: cpuConfig)
        }
    }
    #endif
}
