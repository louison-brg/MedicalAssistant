import SwiftUI

/// Shown once per app process launch. The acknowledgement is intentionally not
/// persisted: reopening the app after it has been terminated shows the notice
/// again, while returning from the background still uses the biometric lock.
struct LaunchGateView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var didAcknowledgeDisclaimer: Bool

    init() {
        let isTesting = ProcessInfo.processInfo.arguments.contains("-uitesting") || ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
        _didAcknowledgeDisclaimer = State(initialValue: isTesting)
    }

    var body: some View {
        Group {
            if didAcknowledgeDisclaimer {
                ContentView()
                    .transition(.opacity)
            } else {
                MedicalDisclaimerView {
                    HapticsHelper.playLightImpact()
                    withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.28)) {
                        didAcknowledgeDisclaimer = true
                    }
                }
                .transition(.opacity)
            }
        }
        .preferredColorScheme(.dark)
    }
}

struct MedicalDisclaimerView: View {
    var isDismissable: Bool = false
    let onContinue: () -> Void

    var body: some View {
        ZStack {
            ChatBackground()
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 24) {
                    if isDismissable {
                        HStack {
                            Spacer()
                            Button {
                                onContinue()
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 24))
                                    .foregroundStyle(.white.opacity(0.6))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Fermer")
                        }
                    }

                    hero

                    VStack(spacing: 12) {
                        noticeCard(
                            icon: "cross.case.fill",
                            tint: ChatTheme.amber,
                            title: "Un assistant, pas un médecin",
                            text: "Les réponses sont générées par une intelligence artificielle locale. Elles peuvent être incomplètes, imprécises ou inexactes et ne constituent ni un diagnostic, ni une prescription, ni un avis médical."
                        )

                        emergencyCard

                        privacyCard
                    }

                    acknowledgement

                    Button(action: onContinue) {
                        HStack(spacing: 10) {
                            Text("J'ai compris — Continuer")
                                .font(.system(.body, design: .rounded, weight: .bold))
                            Image(systemName: "arrow.right")
                                .font(.system(size: 14, weight: .bold))
                        }
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(ChatTheme.accentGradient(load: 0), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .strokeBorder(Color.white.opacity(0.22), lineWidth: 0.6)
                        )
                        .shadow(color: ChatTheme.electricBlue.opacity(0.35), radius: 18, y: 6)
                    }
                    .buttonStyle(.plain)
                    .accessibilityHint("Ouvre l'assistant médical")
                }
                .frame(maxWidth: 560)
                .padding(.horizontal, 20)
                .padding(.top, 28)
                .padding(.bottom, 32)
                .frame(maxWidth: .infinity)
            }
            .scrollIndicators(.hidden)
        }
    }

    private var hero: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                ChatTheme.auroraCyan.opacity(0.22),
                                ChatTheme.electricBlue.opacity(0.08),
                                .clear
                            ],
                            center: .center,
                            startRadius: 15,
                            endRadius: 65
                        )
                    )
                    .frame(width: 120, height: 120)

                Circle()
                    .strokeBorder(
                        LinearGradient(
                            colors: [Color.white.opacity(0.3), ChatTheme.auroraCyan.opacity(0.2), .clear],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
                    .frame(width: 86, height: 86)

                Circle()
                    .fill(
                        LinearGradient(
                            colors: [ChatTheme.auroraCyan.opacity(0.2), ChatTheme.electricBlue.opacity(0.12)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 76, height: 76)

                Image(systemName: "heart.text.clipboard.fill")
                    .font(.system(size: 36, weight: .light))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [ChatTheme.auroraCyan, ChatTheme.sky],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
            }
            .shadow(color: ChatTheme.electricBlue.opacity(0.32), radius: 24)

            VStack(spacing: 7) {
                Text("Avant de commencer")
                    .font(.system(.largeTitle, design: .rounded, weight: .bold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .accessibilityAddTraits(.isHeader)

                Text("Informations importantes concernant MedLLM")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.white.opacity(0.55))
                    .multilineTextAlignment(.center)
            }
        }
    }

    private func noticeCard(icon: String, tint: Color, title: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 13, style: .continuous)
                    .fill(tint.opacity(0.15))
                RoundedRectangle(cornerRadius: 13, style: .continuous)
                    .strokeBorder(tint.opacity(0.3), lineWidth: 0.6)
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(tint)
            }
            .frame(width: 40, height: 40)

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.system(.headline, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.95))

                Text(text)
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.white.opacity(0.62))
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(Color.white.opacity(0.025))
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 0.6)
        )
    }

    private var emergencyCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(ChatTheme.coral.opacity(0.15))
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .strokeBorder(ChatTheme.coral.opacity(0.3), lineWidth: 0.6)
                    Image(systemName: "phone.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(ChatTheme.coral)
                }
                .frame(width: 40, height: 40)

                VStack(alignment: .leading, spacing: 6) {
                    Text("En cas d'urgence médicale")
                        .font(.system(.headline, design: .rounded, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.95))

                    Text("En cas de détresse respiratoire, douleur thoracique aiguë ou doute vital, contactez immédiatement les secours.")
                        .font(.system(.subheadline, design: .rounded))
                        .foregroundStyle(.white.opacity(0.62))
                        .lineSpacing(3)
                }

                Spacer(minLength: 0)
            }

            // Quick dial action buttons
            HStack(spacing: 10) {
                if let url15 = URL(string: "tel://15") {
                    Link(destination: url15) {
                        HStack(spacing: 6) {
                            Image(systemName: "phone.circle.fill")
                            Text("15 · SAMU")
                                .fontWeight(.bold)
                        }
                        .font(.system(.subheadline, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(ChatTheme.coral.opacity(0.85), in: Capsule())
                    }
                }

                if let url112 = URL(string: "tel://112") {
                    Link(destination: url112) {
                        HStack(spacing: 6) {
                            Image(systemName: "phone.circle.fill")
                            Text("112 · Urgence Europe")
                                .fontWeight(.bold)
                        }
                        .font(.system(.subheadline, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(ChatTheme.deepViolet.opacity(0.85), in: Capsule())
                    }
                }
            }
            .padding(.leading, 54)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(ChatTheme.coral.opacity(0.04))
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(ChatTheme.coral.opacity(0.2), lineWidth: 0.6)
        )
    }

    private var privacyCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(ChatTheme.emerald.opacity(0.15))
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .strokeBorder(ChatTheme.emerald.opacity(0.3), lineWidth: 0.6)
                    Image(systemName: "shield.checkered")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(ChatTheme.emerald)
                }
                .frame(width: 40, height: 40)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Confidentialité totale & locale")
                        .font(.system(.headline, design: .rounded, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.95))
                    Text("Aucune donnée ne quitte votre appareil")
                        .font(.system(.caption, design: .rounded, weight: .semibold))
                        .foregroundStyle(ChatTheme.mint)
                }
            }

            VStack(alignment: .leading, spacing: 9) {
                privacyPoint("Le modèle d'IA s'exécute 100% sur le processeur neuronal local.")
                privacyPoint("Vos échanges et questions restent uniquement sur cet iPhone/iPad.")
                privacyPoint("Aucun serveur distant, aucun tracking, aucun compte requis.")
                privacyPoint("Fonctionne même en mode avion, sans aucune connexion Internet.")
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(ChatTheme.emerald.opacity(0.07))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(ChatTheme.emerald.opacity(0.25), lineWidth: 0.7)
        )
    }

    private func privacyPoint(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(ChatTheme.emerald)
                .padding(.top, 2)
            Text(text)
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white.opacity(0.7))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var acknowledgement: some View {
        Text("En continuant, vous confirmez avoir compris ces limites et acceptez d'utiliser MedLLM à des fins d'information personnelle uniquement.")
            .font(.system(.caption, design: .rounded))
            .foregroundStyle(.white.opacity(0.42))
            .multilineTextAlignment(.center)
            .lineSpacing(2.5)
            .padding(.horizontal, 12)
    }
}

#Preview("Avertissement médical") {
    MedicalDisclaimerView(onContinue: {})
        .preferredColorScheme(.dark)
}
