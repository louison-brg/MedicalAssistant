import SwiftUI

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            Text(message.text)
                .padding(12)
                .background(message.isUser ? Color.blue.opacity(0.85) : Color.gray.opacity(0.25))
                .foregroundColor(message.isUser ? .white : .primary)
                .cornerRadius(16)
                .frame(maxWidth: 280, alignment: message.isUser ? .trailing : .leading)
                .padding(message.isUser ? .leading : .trailing, 40)
                .padding(.vertical, 4)
            
            if !message.isUser {
                Spacer()
            }
        }
        .animation(.easeInOut, value: message.text)
    }
}
