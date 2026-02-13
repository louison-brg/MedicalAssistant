import SwiftUI

/// Animated three-dot "thinking" indicator — bounces with staggered phase.
struct TypingIndicator: View {
    @State private var animating = false

    private let dotSize: CGFloat = 7
    private let spacing: CGFloat = 4
    private let bounceDelta: CGFloat = -6

    var body: some View {
        HStack(spacing: spacing) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Color.secondary.opacity(0.55))
                    .frame(width: dotSize, height: dotSize)
                    .offset(y: animating ? bounceDelta : 0)
                    .animation(
                        .easeInOut(duration: 0.45)
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.15),
                        value: animating
                    )
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 8)
        .onAppear { animating = true }
    }
}

#Preview {
    TypingIndicator()
        .padding()
        .background(Color(.systemBackground))
}
