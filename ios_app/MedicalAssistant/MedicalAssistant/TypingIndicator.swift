import SwiftUI

/// NPU "thinking" indicator — breathing glow orb with pulsing ring.
struct TypingIndicator: View {
    @State private var isPulsing = false

    private let orbSize: CGFloat = 22
    private let ringMaxScale: CGFloat = 2.2
    private let glowColor = ChatTheme.electricBlue

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                // Expanding ring
                Circle()
                    .stroke(glowColor.opacity(0.25), lineWidth: 1.5)
                    .frame(width: orbSize, height: orbSize)
                    .scaleEffect(isPulsing ? ringMaxScale : 1.0)
                    .opacity(isPulsing ? 0 : 0.6)

                // Outer glow
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [glowColor.opacity(0.45), glowColor.opacity(0.0)],
                            center: .center,
                            startRadius: 0,
                            endRadius: orbSize * 0.8
                        )
                    )
                    .frame(width: orbSize * 1.6, height: orbSize * 1.6)
                    .scaleEffect(isPulsing ? 1.15 : 0.85)
                    .opacity(isPulsing ? 0.7 : 0.3)

                // Core orb
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [glowColor.opacity(0.9), glowColor.opacity(0.4)],
                            center: .center,
                            startRadius: 0,
                            endRadius: orbSize * 0.5
                        )
                    )
                    .frame(width: orbSize, height: orbSize)
                    .scaleEffect(isPulsing ? 1.1 : 0.9)
            }

            Text("Thinking…")
                .font(.system(.caption, design: .rounded))
                .foregroundStyle(.white.opacity(0.5))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 10)
        .onAppear {
            withAnimation(
                .easeInOut(duration: 1.8)
                .repeatForever(autoreverses: true)
            ) {
                isPulsing = true
            }
        }
    }
}

#Preview {
    ZStack {
        Color(red: 0.06, green: 0.06, blue: 0.10).ignoresSafeArea()
        TypingIndicator()
    }
}
