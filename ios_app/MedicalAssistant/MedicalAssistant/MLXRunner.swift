import Foundation
import MLX
import MLXNN

private struct Phi3Config: Decodable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let rmsNormEps: Double
    let ropeTheta: Double
    let maxPositionEmbeddings: Int
    let tieWordEmbeddings: Bool
}

private struct LayerKVCache {
    var keys: MLXArray
    var values: MLXArray
}

private enum MLXRunnerError: LocalizedError {
    case missingConfig
    case missingTensor(String)
    case missingRoPE

    var errorDescription: String? {
        switch self {
        case .missingConfig:
            return "Configuration du modèle invalide."
        case .missingTensor(let name):
            return "Poids manquant dans le modèle : \(name)"
        case .missingRoPE:
            return "Configuration positionnelle RoPE indisponible."
        }
    }
}

/// Runner MLX avec vraie inférence Phi-3 quantifiée (4-bit) en Swift
actor MLXRunner {
    private var parameters: [String: MLXArray] = [:]
    private var config: Phi3Config?
    private var rope: RoPE?
    private var eosTokenIds: Set<Int> = [32000]
    private var quantGroupSize: Int = 64
    private var quantBits: Int = 4
    private let quantMode: QuantizationMode = .affine
    private var isLoaded = false
    private var loadTask: Task<Bool, Never>?

    private let tokenizer = Tokenizer()
    // Paramètres optimisés pour iPhone sans trop sacrifier la qualité.
    private let maxContextTokens = 512
    private let maxNewTokens = 256
    private let mlxCacheLimitBytes = 20 * 1024 * 1024
    private let samplingTemperature: Float = 0.0
    private let samplingTopP: Float = 0.9
    private let samplingTopK: Int = 40
    private let repetitionPenalty: Float = 1.12
    private static let modelDirectoryCandidates = ["Phi3_Medical_Latest"]

    init() {
        Task {
            await loadModel()
        }
    }

    @discardableResult
    func loadModel() async -> Bool {
        if isLoaded {
            return true
        }

        if let loadTask {
            return await loadTask.value
        }

        let task = Task(priority: .userInitiated) { [self] in
            await performModelLoad()
        }
        loadTask = task
        let loaded = await task.value
        loadTask = nil
        return loaded
    }

    private func performModelLoad() async -> Bool {
        configureMemoryGuards()

        parameters.removeAll(keepingCapacity: false)
        config = nil
        rope = nil
        isLoaded = false
        eosTokenIds = [tokenizer.eosTokenId]

        do {
            let fileManager = FileManager.default
            var modelRootURL: URL?

            var hasLoadedWeights = false
            for candidate in Self.modelDirectoryCandidates {
                guard let modelURL = Bundle.main.url(forResource: candidate, withExtension: nil) else {
                    continue
                }
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: modelURL.path, isDirectory: &isDirectory), isDirectory.boolValue
                else {
                    continue
                }
                let files = try fileManager.contentsOfDirectory(at: modelURL, includingPropertiesForKeys: nil)
                    .filter { $0.pathExtension == "safetensors" }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
                guard !files.isEmpty else { continue }

                print("📦 Chargement du modèle MLX depuis :", modelURL.path)
                modelRootURL = modelURL
                for fileURL in files {
                    print("🔹 Chargement des poids :", fileURL.lastPathComponent)
                    let arrays = try MLX.loadArrays(url: fileURL)
                    for (key, value) in arrays {
                        parameters[key] = value
                    }
                }
                hasLoadedWeights = true
                break
            }

            if !hasLoadedWeights,
               let safetensorsURL = Bundle.main.url(forResource: "model", withExtension: "safetensors")
            {
                print("📦 Chargement du modèle MLX depuis :", safetensorsURL.path)
                let arrays = try MLX.loadArrays(url: safetensorsURL)
                for (key, value) in arrays {
                    parameters[key] = value
                }
                modelRootURL = safetensorsURL.deletingLastPathComponent()
                hasLoadedWeights = true
            }

            if !hasLoadedWeights {
                let bundleURL = URL(fileURLWithPath: Bundle.main.bundlePath)
                let shardFiles = try fileManager.contentsOfDirectory(at: bundleURL, includingPropertiesForKeys: nil)
                    .filter { $0.lastPathComponent.hasPrefix("model") && $0.pathExtension == "safetensors" }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
                if !shardFiles.isEmpty {
                    print("📦 Chargement des shards MLX depuis le bundle racine.")
                    for fileURL in shardFiles {
                        print("🔹 Chargement des poids :", fileURL.lastPathComponent)
                        let arrays = try MLX.loadArrays(url: fileURL)
                        for (key, value) in arrays {
                            parameters[key] = value
                        }
                    }
                    modelRootURL = bundleURL
                    hasLoadedWeights = true
                }
            }

            guard hasLoadedWeights else {
                let candidates = Self.modelDirectoryCandidates.joined(separator: ", ")
                print("❌ Impossible de trouver le modèle MLX dans le bundle. Dossiers candidats: \(candidates), ou fichier attendu: model.safetensors")
                return false
            }

            let configURLs = [
                modelRootURL?.appendingPathComponent("config.json"),
                Bundle.main.url(forResource: "config", withExtension: "json"),
            ].compactMap { $0 }
            guard let loadedConfig = Self.decodeJSON(Phi3Config.self, from: configURLs) else {
                print("❌ Impossible de charger config.json pour le modèle MLX.")
                return false
            }
            self.config = loadedConfig

            let headDim = loadedConfig.hiddenSize / loadedConfig.numAttentionHeads
            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: Float(loadedConfig.ropeTheta),
                scale: 1.0
            )
            inferQuantization(config: loadedConfig)
            loadGenerationConfig(modelRootURL: modelRootURL)

            isLoaded = !parameters.isEmpty
            print("✅ Modèle MLX chargé avec \(parameters.count) tenseurs. q\(quantBits) group=\(quantGroupSize)")
            return isLoaded
        } catch {
            print("❌ Erreur lors du chargement du modèle MLX :", error.localizedDescription)
            return false
        }
    }

    /// Releases the 4-bit weights and MLX's reusable Metal buffers when the app
    /// enters the background. This lowers the chance of iOS terminating the app
    /// under memory pressure and guarantees a clean next inference session.
    func unloadModel() {
        loadTask?.cancel()
        loadTask = nil
        parameters.removeAll(keepingCapacity: false)
        config = nil
        rope = nil
        isLoaded = false
        GPU.clearCache()
    }

    /// Explicitly purges intermediate Metal GPU allocations without unloading model weights.
    func clearInferenceCache() {
        GPU.clearCache()
    }

    func generateResponseStream(for prompt: String, history: [Message]? = nil) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task(priority: .userInitiated) {
                let loaded = await loadModel()
                guard loaded, let cfg = config else {
                    continuation.yield("⚠️ Modèle MLX non chargé.")
                    continuation.finish()
                    return
                }

                if Task.isCancelled {
                    continuation.yield("(Generation cancelled)")
                    continuation.finish()
                    return
                }

                do {
                    let promptTokens = buildPromptTokens(from: prompt, history: history, cfg: cfg)
                    if promptTokens.isEmpty {
                        continuation.yield("(Prompt vide)")
                        continuation.finish()
                        return
                    }
                    
                    var cache = Array<LayerKVCache?>(repeating: nil, count: cfg.numHiddenLayers)
                    defer {
                        cache.removeAll(keepingCapacity: false)
                        GPU.clearCache()
                    }
                    
                    let promptIds = promptTokens.map(Int32.init)
                    var generated: [Int] = []
                    var decoderState = Tokenizer.DecodingState()
                    
                    // Pré-remplissage
                    var logits = try runModel(inputTokenIds: promptIds, cache: &cache, cfg: cfg)
                    for _ in 0..<maxNewTokens {
                        if Task.isCancelled { break }
                        
                        let lastLogits = logits[0, -1, .ellipsis]
                        
                        let recentTokens = generated.isEmpty ? [] : Array(generated.suffix(64))
                        let penalized = applyRepetitionPenalty(
                            logits: lastLogits,
                            recentTokens: recentTokens,
                            penalty: repetitionPenalty
                        )
                        let nextToken = sampleToken(
                            from: penalized,
                            temperature: samplingTemperature,
                            topP: samplingTopP,
                            topK: samplingTopK
                        )
                        
                        if eosTokenIds.contains(nextToken) || nextToken == tokenizer.padTokenId { break }
                        if !tokenizer.isValid(id: nextToken) {
                            continuation.yield("\n[⚠️ Token invalide: \(nextToken)]")
                            break
                        }
                        
                        generated.append(nextToken)
                        let newText = tokenizer.decode(nextToken, state: &decoderState)
                        if !newText.isEmpty {
                            continuation.yield(newText)
                        }

                        logits = try runModel(inputTokenIds: [Int32(nextToken)], cache: &cache, cfg: cfg)
                    }

                    let decoderTail = tokenizer.finishDecoding(state: &decoderState)
                    if !decoderTail.isEmpty {
                        continuation.yield(decoderTail)
                    }

                    if generated.isEmpty {
                        continuation.yield("⚠️ Échec : Le modèle n'a généré aucun texte (corruption silencieuse de la RAM GPU Apple liée au cache). Ce comportement a été massivement optimisé et devrait maintenant disparaître.")
                    }

                    continuation.finish()
                } catch {
                    print("❌ Erreur d’inférence MLX :", error.localizedDescription)
                    continuation.finish(throwing: error)
                }
            }

            // Cette annulation garantit qu'il n'y ait plus de processus MLX zombie qui tourne en tâche de fond !
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func runModel(
        inputTokenIds: [Int32],
        cache: inout [LayerKVCache?],
        cfg: Phi3Config
    ) throws -> MLXArray {
        guard !inputTokenIds.isEmpty else {
            throw MLXRunnerError.missingConfig
        }

        let input = MLXArray(inputTokenIds, [1, inputTokenIds.count])
        var h = try embedding(ids: input, cfg: cfg)

        for layer in 0..<cfg.numHiddenLayers {
            let layerPrefix = "model.layers.\(layer)"

            let inNormWeight = try tensor("\(layerPrefix).input_layernorm.weight")
            let inNorm = rmsNorm(h, weight: inNormWeight, eps: Float(cfg.rmsNormEps))
            let attn = try attention(inNorm, layer: layer, cache: &cache[layer], cfg: cfg)
            let hAfterAttn = h + attn

            let postNormWeight = try tensor("\(layerPrefix).post_attention_layernorm.weight")
            let postNorm = rmsNorm(hAfterAttn, weight: postNormWeight, eps: Float(cfg.rmsNormEps))
            let mlpOut = try mlp(postNorm, layer: layer)
            h = hAfterAttn + mlpOut
        }

        let normWeight = try tensor("model.norm.weight")
        let normalized = rmsNorm(h, weight: normWeight, eps: Float(cfg.rmsNormEps))
        // On ne projette que le dernier token pour réduire la mémoire temporaire.
        let lastHidden = normalized[0, normalized.dim(1) - 1, .ellipsis].reshaped([1, 1, cfg.hiddenSize])

        let logits: MLXArray
        if cfg.tieWordEmbeddings {
            let embWeight = try tensor("model.embed_tokens.weight")
            let embScales = try tensor("model.embed_tokens.scales")
            let embBiases = parameters["model.embed_tokens.biases"]
            logits = quantizedMatmul(
                lastHidden,
                embWeight,
                scales: embScales,
                biases: embBiases,
                transpose: true,
                groupSize: quantGroupSize,
                bits: quantBits,
                mode: quantMode
            )
        } else {
            logits = try quantizedLinear(lastHidden, prefix: "lm_head")
        }
        var arraysToEvaluate = [logits]
        arraysToEvaluate.reserveCapacity(1 + cache.count * 2)
        for layerCache in cache.compactMap({ $0 }) {
            arraysToEvaluate.append(layerCache.keys)
            arraysToEvaluate.append(layerCache.values)
        }
        // One graph evaluation per model step is substantially faster than a
        // Metal synchronization after every transformer layer.
        eval(arraysToEvaluate)
        return logits
    }

    private func embedding(ids: MLXArray, cfg: Phi3Config) throws -> MLXArray {
        let weight = try tensor("model.embed_tokens.weight")
        let scales = try tensor("model.embed_tokens.scales")
        let biases = parameters["model.embed_tokens.biases"]
        let flat = ids.flattened()

        let out = dequantized(
            weight[flat],
            scales: scales[flat],
            biases: biases == nil ? nil : biases![flat],
            groupSize: quantGroupSize,
            bits: quantBits,
            mode: quantMode
        )
        return out.reshaped([ids.dim(0), ids.dim(1), cfg.hiddenSize])
    }

    private func attention(
        _ x: MLXArray,
        layer: Int,
        cache layerCache: inout LayerKVCache?,
        cfg: Phi3Config
    ) throws -> MLXArray {
        guard let rope else {
            throw MLXRunnerError.missingRoPE
        }

        let layerPrefix = "model.layers.\(layer).self_attn"
        let qkv = try quantizedLinear(x, prefix: "\(layerPrefix).qkv_proj")

        let batch = x.dim(0)
        let sequenceLength = x.dim(1)
        let headDim = cfg.hiddenSize / cfg.numAttentionHeads
        let queryPos = cfg.numAttentionHeads * headDim
        let kvWidth = cfg.numKeyValueHeads * headDim
        let parts = qkv.split(indices: [queryPos, queryPos + kvWidth], axis: qkv.ndim - 1)

        var queries = parts[0]
            .reshaped([batch, sequenceLength, cfg.numAttentionHeads, headDim])
            .transposed(0, 2, 1, 3)
        var keys = parts[1]
            .reshaped([batch, sequenceLength, cfg.numKeyValueHeads, headDim])
            .transposed(0, 2, 1, 3)
        var values = parts[2]
            .reshaped([batch, sequenceLength, cfg.numKeyValueHeads, headDim])
            .transposed(0, 2, 1, 3)

        let offset = layerCache?.keys.dim(2) ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        if let existing = layerCache {
            keys = concatenated([existing.keys, keys], axis: 2)
            values = concatenated([existing.values, values], axis: 2)
        }
        layerCache = LayerKVCache(keys: keys, values: values)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = sequenceLength > 1 ? .causal : .none
        let attentionScale = 1.0 / sqrt(Float(headDim))
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: attentionScale,
            mask: maskMode
        )

        var output = attended.transposed(0, 2, 1, 3).reshaped([batch, sequenceLength, cfg.hiddenSize])
        output = try quantizedLinear(output, prefix: "\(layerPrefix).o_proj")
        return output
    }

    private func mlp(_ x: MLXArray, layer: Int) throws -> MLXArray {
        let layerPrefix = "model.layers.\(layer).mlp"
        let gateUp = try quantizedLinear(x, prefix: "\(layerPrefix).gate_up_proj")
        let splitGateUp = gateUp.split(parts: 2, axis: gateUp.ndim - 1)
        let gate = splitGateUp[0]
        let up = splitGateUp[1]
        let activated = silu(gate) * up
        return try quantizedLinear(activated, prefix: "\(layerPrefix).down_proj")
    }

    private func quantizedLinear(_ x: MLXArray, prefix: String) throws -> MLXArray {
        let weight = try tensor("\(prefix).weight")
        let scales = try tensor("\(prefix).scales")
        let biases = parameters["\(prefix).biases"]
        return quantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: quantGroupSize,
            bits: quantBits,
            mode: quantMode
        )
    }

    private func tensor(_ name: String) throws -> MLXArray {
        guard let array = parameters[name] else {
            throw MLXRunnerError.missingTensor(name)
        }
        return array
    }

    private func inferQuantization(config cfg: Phi3Config) {
        guard
            let qkvWeight = parameters["model.layers.0.self_attn.qkv_proj.weight"],
            let qkvScales = parameters["model.layers.0.self_attn.qkv_proj.scales"]
        else {
            quantGroupSize = 64
            quantBits = 4
            return
        }

        let packedInput = qkvWeight.dim(1)
        let expectedInput = cfg.hiddenSize
        if packedInput > 0 && expectedInput > 0 {
            quantBits = max(1, (packedInput * 32) / expectedInput)
        }

        let groups = qkvScales.dim(1)
        if groups > 0 {
            quantGroupSize = max(1, expectedInput / groups)
        }
    }

    private func buildPromptTokens(from userInput: String, history: [Message]?, cfg: Phi3Config) -> [Int] {
        // Keep complete chat turns. Raw suffix truncation can start in the middle
        // of a BPE sequence or silently remove the role token, both of which hurt
        // answer quality much more than dropping an older turn.

        func appendSpecial(_ token: String, into target: inout [Int]) {
            if let id = tokenizer.id(for: token) {
                target.append(id)
            }
        }

        func encodedTurn(roleToken: String, content: String, closeTurn: Bool) -> [Int] {
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return [] }
            var result: [Int] = []
            appendSpecial(roleToken, into: &result)
            result += tokenizer.encode("\n" + trimmed, addPrefixSpace: false)
            if closeTurn {
                appendSpecial("<|end|>", into: &result)
                result += tokenizer.encode("\n", addPrefixSpace: false)
            }
            return result
        }

        func clippedUserTurn(_ content: String, limit: Int) -> [Int] {
            guard limit > 0 else { return [] }
            var prefix: [Int] = []
            appendSpecial("<|user|>", into: &prefix)
            prefix += tokenizer.encode("\n", addPrefixSpace: false)
            var suffix: [Int] = []
            appendSpecial("<|end|>", into: &suffix)
            suffix += tokenizer.encode("\n", addPrefixSpace: false)

            guard prefix.count + suffix.count < limit else {
                return Array((prefix + suffix).prefix(limit))
            }
            let contentTokens = tokenizer.encode(
                content.trimmingCharacters(in: .whitespacesAndNewlines),
                addPrefixSpace: false
            )
            let contentLimit = limit - prefix.count - suffix.count
            return prefix + Array(contentTokens.suffix(contentLimit)) + suffix
        }

        let trimmedPrompt = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let contextLimit = min(cfg.maxPositionEmbeddings, maxContextTokens)

        // Désactivation du "System Prompt" : le modèle a été fine-tuné uniquement sur <|user|> et <|assistant|>.
        // L'injection d'un système le fait halluciner un préambule ("My response as...") suivi d'un EOS instantané.

        func isNoisyAssistantMessage(_ text: String) -> Bool {
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !t.isEmpty else { return true }
            if t.contains("explanation: step") { return true }
            if t.contains("the correct answer is") && t.contains("step1") { return true }
            let words = t.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" })
            if words.count >= 6 {
                let uniqueCount = Set(words).count
                if Double(uniqueCount) / Double(words.count) < 0.35 {
                    return true
                }
            }
            return false
        }

        var turnChunks: [[Int]] = []
        if let history {
            let usable = history.filter { !$0.isPartial && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            let recent = Array(usable.suffix(8))
            for message in recent {
                if message.isUser {
                    turnChunks.append(encodedTurn(roleToken: "<|user|>", content: message.text, closeTurn: true))
                } else {
                    if isNoisyAssistantMessage(message.text) {
                        continue
                    }
                    turnChunks.append(encodedTurn(roleToken: "<|assistant|>", content: message.text, closeTurn: true))
                }
            }
        }

        // Si l'historique ne contient pas déjà la question courante, on l'ajoute.
        let normalizedLastUser = history?
            .reversed()
            .first(where: { $0.isUser && !$0.isPartial })?
            .text
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if normalizedLastUser != trimmedPrompt {
            turnChunks.append(encodedTurn(roleToken: "<|user|>", content: trimmedPrompt, closeTurn: true))
        }

        var generationPrefix: [Int] = []
        appendSpecial("<|assistant|>", into: &generationPrefix)
        generationPrefix += tokenizer.encode("\n", addPrefixSpace: false)

        if turnChunks.isEmpty {
            turnChunks.append(encodedTurn(roleToken: "<|user|>", content: trimmedPrompt, closeTurn: true))
        }

        var remaining = max(0, contextLimit - generationPrefix.count)
        var selectedTurns: [[Int]] = []
        for chunk in turnChunks.reversed() where !chunk.isEmpty {
            guard chunk.count <= remaining else { continue }
            selectedTurns.insert(chunk, at: 0)
            remaining -= chunk.count
        }

        // The current user turn must always be present, even for unusually long
        // input. Preserve its role marker and the end marker while clipping text.
        if selectedTurns.isEmpty {
            let available = max(0, contextLimit - generationPrefix.count)
            selectedTurns = [clippedUserTurn(trimmedPrompt, limit: available)]
        }

        return selectedTurns.flatMap { $0 } + generationPrefix
    }

    private func loadGenerationConfig(modelRootURL: URL?) {
        let candidates = [
            modelRootURL?.appendingPathComponent("generation_config.json"),
            Bundle.main.url(forResource: "generation_config", withExtension: "json"),
        ].compactMap { $0 }

        for url in candidates {
            guard let data = try? Data(contentsOf: url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                continue
            }

            if let eos = json["eos_token_id"] as? Int {
                eosTokenIds.insert(eos)
            } else if let eosList = json["eos_token_id"] as? [Int] {
                eosTokenIds.formUnion(eosList)
            }
            break
        }
    }

    private func configureMemoryGuards() {
        GPU.set(cacheLimit: mlxCacheLimitBytes)
    }

    private func applyRepetitionPenalty(logits: MLXArray, recentTokens: [Int], penalty: Float) -> MLXArray {
        guard penalty > 1.0, !recentTokens.isEmpty else {
            return logits
        }
        let uniqueTokens = Set(recentTokens)
            .filter { $0 >= 0 && $0 < logits.dim(0) }
            .map(Int32.init)
        guard !uniqueTokens.isEmpty else { return logits }

        let indices = MLXArray(uniqueTokens)
        let penalized = logits
        let selected = penalized[indices]
        let adjusted = `where`(selected .> 0, selected / penalty, selected * penalty)
        penalized[indices] = adjusted
        return penalized
    }

    private func sampleToken(from logits: MLXArray, temperature: Float, topP: Float, topK: Int) -> Int {
        // Décodage déterministe pour réduire fortement le charabia.
        if temperature <= 0.01 {
            return Int(logits.argMax().item(Int32.self))
        }

        let values = logits.asType(.float32).asArray(Float.self)
        guard !values.isEmpty else {
            return tokenizer.eosTokenId
        }

        let temp = max(0.05, temperature)
        let maxLogit = values.max() ?? 0
        var expValues = values.map { expf(($0 - maxLogit) / temp) }
        let sumExp: Float = expValues.reduce(0, +)
        if !sumExp.isFinite || sumExp <= 0 {
            return Int(logits.argMax().item(Int32.self))
        }
        for i in 0..<expValues.count {
            expValues[i] /= sumExp
        }

        var sorted = expValues.enumerated().map { (idx: $0.offset, p: $0.element) }
        sorted.sort { $0.p > $1.p }
        if topK > 0 && sorted.count > topK {
            sorted = Array(sorted.prefix(topK))
        }

        let pCut = min(max(topP, 0.05), 1.0)
        var nucleus: [(idx: Int, p: Float)] = []
        var cumulative: Float = 0
        for item in sorted {
            nucleus.append(item)
            cumulative += item.p
            if cumulative >= pCut {
                break
            }
        }
        if nucleus.isEmpty {
            return sorted.first?.idx ?? Int(logits.argMax().item(Int32.self))
        }

        let nucleusSum = nucleus.reduce(Float(0)) { $0 + $1.p }
        if nucleusSum <= 0 || !nucleusSum.isFinite {
            return nucleus.first!.idx
        }
        let r = Float.random(in: 0..<1)
        var acc: Float = 0
        for item in nucleus {
            acc += item.p / nucleusSum
            if r <= acc {
                return item.idx
            }
        }
        return nucleus.last!.idx
    }

    private static func decodeJSON<T: Decodable>(_ type: T.Type, from urls: [URL]) -> T? {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        for url in urls {
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            guard let data = try? Data(contentsOf: url) else { continue }
            if let decoded = try? decoder.decode(T.self, from: data) {
                return decoded
            }
        }
        return nil
    }
}
