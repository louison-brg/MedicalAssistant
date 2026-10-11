import SwiftUI

/// Elegant animated medical thinking indicator with pulsing wave dots and subtle ambient glow.
struct TypingIndicator: View {
    @State private var phase = 0.0
    @State private var isGlowBreathing = false

    private let dotCount = 3
    private let dotSize: CGFloat = 7
    private let glowColor = ChatTheme.electricBlue
    private let cyanColor = ChatTheme.auroraCyan

    var body: some View {
        HStack(spacing: 11) {
            // Ambient icon
            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                cyanColor.opacity(isGlowBreathing ? 0.35 : 0.15),
                                glowColor.opacity(0.0)
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: 16
                        )
                    )
                    .frame(width: 28, height: 28)
                    .scaleEffect(isGlowBreathing ? 1.2 : 0.9)

                Image(systemName: "sparkles")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [cyanColor, glowColor],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }

            // Pulsing wave dots
            HStack(spacing: 5) {
                ForEach(0..<dotCount, id: \.self) { index in
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [cyanColor, glowColor],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: dotSize, height: dotSize)
                        .scaleEffect(dotScale(for: index))
                        .opacity(dotOpacity(for: index))
                }
            }
            .frame(height: 18)

            Text("Analyzing…")
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.65))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Capsule()
                .fill(.ultraThinMaterial)
                .overlay(
                    Capsule()
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.18), cyanColor.opacity(0.15)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 0.5
                        )
                )
        )
        .shadow(color: glowColor.opacity(0.15), radius: 10, y: 3)
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                isGlowBreathing = true
            }
            withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) {
                phase = 1.0
            }
        }
    }

    private func dotScale(for index: Int) -> CGFloat {
        let progress = (phase + Double(index) / Double(dotCount)).truncatingRemainder(dividingBy: 1.0)
        let sine = sin(progress * .pi)
        return 0.75 + (0.45 * sine)
    }

    private func dotOpacity(for index: Int) -> Double {
        let progress = (phase + Double(index) / Double(dotCount)).truncatingRemainder(dividingBy: 1.0)
        let sine = sin(progress * .pi)
        return 0.35 + (0.65 * sine)
    }
}

#Preview {
    ZStack {
        ChatTheme.backgroundBase.ignoresSafeArea()
        TypingIndicator()
    }
}
