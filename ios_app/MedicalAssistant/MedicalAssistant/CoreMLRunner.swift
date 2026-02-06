import Foundation
import CoreML

struct GenerationConfig {
    var temperature: Double = 0.6
    var topK: Int = 20
    var topP: Double = 0.85
    var repetitionPenalty: Double = 1.01
    var maxNewTokens: Int = 128
    var partialUpdateTokenInterval: Int = 4
}

/// 🧠 Classe utilitaire qui encapsule le modèle CoreML Phi-3
final class CoreMLRunner: @unchecked Sendable {
    private var model: MLModel?
    private let tokenizer = Tokenizer()
    private var preferGPU: Bool = true
    private var computeUnits: MLComputeUnits = .all
    private let isoFormatter = ISO8601DateFormatter()
    private var didDumpLogits = false

    init(preferGPU: Bool = true) {
        self.preferGPU = preferGPU
        loadModel()
    }

    /// Charge le modèle CoreML (.mlpackage compilé en .mlmodelc)
    private func loadModel(computeUnits override: MLComputeUnits? = nil) {
        if let overrideUnits = override {
            self.computeUnits = overrideUnits
        }
        let bundle = Bundle.main
        do {
            if let compiledURL = bundle.url(forResource: "MedicalLLM", withExtension: "mlmodelc") {
                self.model = try CoreMLConfig.loadModel(at: compiledURL, preferGPU: self.preferGPU, computeUnits: self.computeUnits)
                print("✅ Modèle CoreML chargé (mlmodelc) :", compiledURL.lastPathComponent)
                return
            }

            if let packageURL = bundle.url(forResource: "MedicalLLM", withExtension: "mlpackage") {
                let compiledURL = try MLModel.compileModel(at: packageURL)
                try loadCompiledModel(at: compiledURL)
                return
            }

            if let rawURL = bundle.url(forResource: "MedicalLLM", withExtension: "mlmodel") {
                let compiledURL = try MLModel.compileModel(at: rawURL)
                try loadCompiledModel(at: compiledURL)
                return
            }

            print("❌ Impossible de trouver MedicalLLM (.mlmodelc/.mlpackage/.mlmodel) dans le bundle.")
        } catch {
            print("⚠️ Erreur de chargement du modèle :", error.localizedDescription)
        }
    }

    private func loadCompiledModel(at url: URL) throws {
        do {
            self.model = try CoreMLConfig.loadModel(at: url, preferGPU: self.preferGPU, computeUnits: self.computeUnits)
            print("✅ Modèle CoreML compilé :", url.lastPathComponent)
        } catch {
            print("⚠️ Échec sur computeUnits=\(self.computeUnits) — tentative CPU only :", error.localizedDescription)
            do {
                self.model = try CoreMLConfig.loadModel(at: url, preferGPU: false, computeUnits: .cpuOnly)
                self.computeUnits = .cpuOnly
                print("✅ Modèle CoreML rechargé en mode CPU only.")
            } catch {
                self.model = nil
                throw error
            }
        }
    }

    /// Effectue une inférence sur du texte d’entrée
    func generateResponse(
        for text: String,
        config: GenerationConfig = GenerationConfig(),
        onPartial: ((String) -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) -> String {
        guard let model = model else {
            return "⚠️ Modèle non chargé."
        }

        func appendSpecial(_ token: String, into ids: inout [Int]) {
            if let tid = tokenizer.id(for: token) {
                ids.append(tid)
            } else {
                print("⚠️ Token spécial absent du vocab :", token)
            }
        }

        // 1️⃣ Construction du prompt tokenisé (system + user + assistant)
        var tokens: [Int] = []
        appendSpecial("<|system|>", into: &tokens)
        tokens += tokenizer.encode(
            "You are a medical doctor answering patients. Always respond in English, stay strictly on the medical question asked, give short factual bullet points when possible, and never rewrite or spell-check the user's prompt."
        )
        appendSpecial("<|end|>", into: &tokens)
        tokens += tokenizer.encode("\n")
        appendSpecial("<|user|>", into: &tokens)
        tokens += tokenizer.encode(text)
        appendSpecial("<|end|>", into: &tokens)
        tokens += tokenizer.encode("\n")
        appendSpecial("<|assistant|>", into: &tokens)

        // 1b️⃣ Récupérer dynamiquement la longueur attendue si possible
        let inputDesc = model.modelDescription.inputDescriptionsByName["input_ids"]
        let constraint = inputDesc?.multiArrayConstraint
        let seqLen: Int = {
            if let shape = constraint?.shape, shape.count == 2 {
                if let dim = shape.last, dim.intValue > 0 { return dim.intValue }
            } else if let shape = constraint?.shape, shape.count == 1 {
                if let dim = shape.first, dim.intValue > 0 { return dim.intValue }
            }
            return 64
        }()

        if tokens.count > seqLen {
            tokens = Array(tokens.suffix(seqLen))
        }

        let padId = tokenizer.padTokenId
        let eosId = tokenizer.eosTokenId
        let unkId = tokenizer.id(for: "<unk>")
        let maxNewTokens = config.maxNewTokens
        let basePromptTokens = tokens

        func selectValidToken(from distribution: [Double]) -> Int {
            var probs = distribution
            var token = sampleIndex(from: probs)
            var attempts = 0
            while !tokenizer.isValid(id: token) && attempts < 10 {
                probs[token] = 0
                let sum = probs.reduce(0, +)
                if sum <= 0 {
                    break
                }
                for i in 0..<probs.count {
                    probs[i] /= sum
                }
                token = sampleIndex(from: probs)
                attempts += 1
            }
            if !tokenizer.isValid(id: token) {
                return tokenizer.id(for: "<unk>") ?? eosId
            }
            return token
        }

        func cleanedTokens(_ tokens: [Int]) -> [Int] {
            var result = tokens
            while let last = result.last, last == padId || last == eosId {
                result.removeLast()
            }
            return result
        }

        func sampleTokens(temp: Double, topK: Int, topP: Double) -> ([Int], Double) {
            do {
                let inputArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
                let maskArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
                var contextTokens = basePromptTokens
                var generatedTokens: [Int] = []
                var totalPredictionTime: Double = 0
                var lastPartialCount = 0

                generationLoop: for _ in 0..<maxNewTokens {
                    if shouldCancel?() == true {
                        break generationLoop
                    }
                    var windowTokens = contextTokens
                    if windowTokens.count > seqLen {
                        windowTokens = Array(windowTokens.suffix(seqLen))
                    }
                    let padCount = max(0, seqLen - windowTokens.count)

                    for i in 0..<seqLen {
                        if i < padCount {
                            inputArray[i] = NSNumber(value: padId)
                            maskArray[i] = 0
                        } else {
                            let token = windowTokens[i - padCount]
                            inputArray[i] = NSNumber(value: token)
                            maskArray[i] = 1
                        }
                    }

                    var dict: [String: Any] = ["input_ids": inputArray]
                    if model.modelDescription.inputDescriptionsByName.keys.contains("attention_mask") {
                        dict["attention_mask"] = maskArray
                    }

                    let input = try MLDictionaryFeatureProvider(dictionary: dict)
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let output = try model.prediction(from: input)
                    let t1 = CFAbsoluteTimeGetCurrent()
                    totalPredictionTime += (t1 - t0)

                    guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                        break
                    }
                    let shape = logits.shape.compactMap { $0.intValue }
                    guard shape.count == 3 else { break }
                    let vocab = shape[2]
                    let lastIndex = max(0, padCount + min(windowTokens.count, seqLen - padCount) - 1)
                    let base = lastIndex * vocab
                    var stepLogits = [Double](repeating: 0.0, count: vocab)
                    for v in 0..<vocab {
                        stepLogits[v] = logits[base + v].doubleValue
                    }

                    #if DEBUG
                    if !didDumpLogits {
                        dumpDebugLogits(logits: logits, baseIndex: base, vocab: vocab)
                        didDumpLogits = true
                    }
                    #endif

                    // Ne jamais échantillonner immédiatement les tokens interdits
                    let forbidden = [padId, unkId].compactMap { $0 }
                    for idx in forbidden where idx >= 0 && idx < stepLogits.count {
                        stepLogits[idx] = -Double.greatestFiniteMagnitude
                    }

                    if !generatedTokens.isEmpty && config.repetitionPenalty > 1.0 {
                        let recent = Set(generatedTokens.suffix(32))
                        for idx in recent where idx < stepLogits.count {
                            stepLogits[idx] /= config.repetitionPenalty
                        }
                    }

                    let probs = softmax(stepLogits, temperature: temp)
                    let filtered = topKTopPFilter(probs: probs, topK: topK, topP: topP)
                    let nextToken = selectValidToken(from: filtered)
                    contextTokens.append(nextToken)
                    generatedTokens.append(nextToken)
                    if contextTokens.count > seqLen {
                        contextTokens = Array(contextTokens.suffix(seqLen))
                    }

                    if let onPartial = onPartial, generatedTokens.count - lastPartialCount >= max(1, config.partialUpdateTokenInterval) {
                        let partial = tokenizer.decode(cleanedTokens(generatedTokens)).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !partial.isEmpty {
                            onPartial(partial)
                            lastPartialCount = generatedTokens.count
                        }
                    }

                    if nextToken == eosId || nextToken == padId {
                        if generatedTokens.count < 4 {
                            // Ignore sortie trop courte et continue
                            generatedTokens.removeLast()
                            contextTokens.removeLast()
                            continue
                        }
                        break generationLoop
                    }
                }

                #if DEBUG
                print(String(format: "⏱️ CoreML prediction took %.2f ms", totalPredictionTime * 1000))
                #endif
                return (generatedTokens, totalPredictionTime)
            } catch {
                print("❌ Erreur d’inférence :", error.localizedDescription)
                return ([], 0)
            }
        }

        let (primaryTokens, _) = sampleTokens(temp: config.temperature, topK: config.topK, topP: config.topP)
        var filteredTokens = cleanedTokens(primaryTokens)

        if filteredTokens.isEmpty {
            print("⚠️ Ré-échantillonnage avec paramètres fallback (temp=1.0, topK=50, topP=0.98)")
            let (retryTokens, _) = sampleTokens(temp: 1.0, topK: 50, topP: 0.98)
            filteredTokens = cleanedTokens(retryTokens)
        }
        #if DEBUG
        print("🔤 Tokens générés:", filteredTokens.map(String.init).joined(separator: ","))
        #endif

        guard !filteredTokens.isEmpty else {
            return "(Pas de réponse générée — réessayez)"
        }

        let rawDecoded = tokenizer.decode(filteredTokens)
        let trimmed = rawDecoded.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            onPartial?(trimmed)
            return trimmed
        }
        if !rawDecoded.isEmpty {
            onPartial?(rawDecoded)
            return rawDecoded
        }

        return "(Réponse indisponible pour cette requête)"
    }

    private func dumpDebugLogits(logits: MLMultiArray, baseIndex: Int, vocab: Int) {
        guard let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        var values: [Double] = []
        values.reserveCapacity(vocab)
        for v in 0..<vocab {
            values.append(logits[baseIndex + v].doubleValue)
        }
        let payload: [String: Any] = [
            "timestamp": isoFormatter.string(from: Date()),
            "values": values
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) {
            let url = docsURL.appendingPathComponent("logits_coreml.json")
            try? data.write(to: url)
            print("📝 Logits dumpés dans :", url.path)
        }
    }

// MARK: - Sampling utilities
    private func softmax(_ logits: [Double], temperature: Double) -> [Double] {
        let invTemp = max(1e-6, temperature)
        let scaled = logits.map { $0 / invTemp }
        let maxLogit = scaled.max() ?? 0
        let exps = scaled.map { exp($0 - maxLogit) }
        let sumExp = exps.reduce(0, +)
        if sumExp == 0 { return Array(repeating: 1.0 / Double(max(1, logits.count)), count: logits.count) }
        return exps.map { $0 / sumExp }
    }

    private func topKTopPFilter(probs: [Double], topK: Int, topP: Double) -> [Double] {
        // Sort indices by probability descending
        let indexed = probs.enumerated().sorted { $0.element > $1.element }
        var kept: [(Int, Double)] = []
        var cumulative: Double = 0
        let k = max(1, topK)
        for (i, pair) in indexed.enumerated() {
            if i < k { kept.append(pair); cumulative += pair.element; continue }
            if cumulative < topP { kept.append(pair); cumulative += pair.element } else { break }
        }
        // Renormalize
        var filtered = Array(repeating: 0.0, count: probs.count)
        let sumKept = kept.reduce(0) { $0 + $1.1 }
        if sumKept == 0 { return probs }
        for (idx, p) in kept { filtered[idx] = p / sumKept }
        return filtered
    }

    private func sampleIndex(from probs: [Double]) -> Int {
        let r = Double.random(in: 0..<1)
        var acc = 0.0
        for (i, p) in probs.enumerated() {
            acc += p
            if r <= acc { return i }
        }
        return probs.indices.last ?? 0
    }

    /// Bascule le mode de performance et recharge le modèle
    func setPerformance(preferGPU: Bool) {
        self.preferGPU = preferGPU
        print("⚙️ Mode performance:", preferGPU ? "GPU (fallback CPU)" : "CPU only")
        loadModel()
    }

    /// Force un computeUnit particulier (CPU only, GPU only, etc.)
    func setComputeUnits(_ units: MLComputeUnits) {
        print("⚙️ Changement computeUnits ->", units)
        loadModel(computeUnits: units)
    }
}
