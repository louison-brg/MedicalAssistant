import Foundation
#if canImport(CoreHaptics)
import CoreHaptics
#endif

enum HapticsHelper {
    /// Indique si les haptics sont supportés par le matériel.
    static var isSupported: Bool {
        #if canImport(CoreHaptics)
        return CHHapticEngine.capabilitiesForHardware().supportsHaptics
        #else
        return false
        #endif
    }

    /// Joue un feedback simple si supporté, sinon ne fait rien.
    static func playLightImpact() {
        #if canImport(CoreHaptics)
        guard isSupported else { return }
        do {
            let engine = try CHHapticEngine()
            try engine.start()

            let event = CHHapticEvent(eventType: .hapticTransient,
                                      parameters: [
                                          CHHapticEventParameter(parameterID: .hapticIntensity, value: 0.4),
                                          CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.4)
                                      ],
                                      relativeTime: 0)
            let pattern = try CHHapticPattern(events: [event], parameters: [])
            let player = try engine.makePlayer(with: pattern)
            try player.start(atTime: 0)
            engine.stop(completionHandler: nil)
        } catch {
            // Silence les erreurs d'environnement (simulateur, macOS sans Taptic Engine, etc.)
        }
        #endif
    }
}
