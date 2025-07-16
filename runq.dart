import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:args/args.dart';

// Глобальная переменная для размера группы квантования.
late int GS;

// ----------------------------------------------------------------------------
// Структуры Transformer
// ----------------------------------------------------------------------------
class Config {
  final int magicNumber, version, dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, headDim, groupSize;
  int seqLen;
  final bool sharedClassifier;
  Config({
    required this.magicNumber,
    required this.version,
    required this.dim,
    required this.hiddenDim,
    required this.nLayers,
    required this.nHeads,
    required this.nKVHeads,
    required this.vocabSize,
    required this.seqLen,
    required this.headDim,
    required this.sharedClassifier,
    required this.groupSize
  });
  @override String toString() => 'hidden_size=$dim, intermediate_size=$hiddenDim, num_hidden_layers=$nLayers, num_attention_heads=$nHeads, num_kv_heads=$nKVHeads, head_dim=$headDim, ctx_length=$seqLen, vocab_size=$vocabSize, shared_classifier=${sharedClassifier ? 1 : 0}, quantization_block_size=$groupSize';
}

class QuantizedTensor {
  Int8List q;
  Float32List s;
  QuantizedTensor({required this.q, required this.s});
  factory QuantizedTensor.empty(int qSize, int sSize) => QuantizedTensor(q: Int8List(qSize), s: Float32List(sSize));
}

class TransformerWeights {
  late QuantizedTensor qTokens;
  late Float32List tokenEmbeddingTable, rmsAttWeight, rmsFfnWeight, qlnWeights, klnWeights, rmsFinalWeight;
  late List<QuantizedTensor> wq, wk, wv, wo, w1, w2, w3;
  late QuantizedTensor wCls;
}

class RunState {
  final Float32List x, xb, xb2, hb, hb2, q, att, logits, keyCache, valueCache;
  final QuantizedTensor xq, hq;
  
  RunState(Config p) :
    x = Float32List(p.dim),
    xb = Float32List(p.nHeads * p.headDim),
    xb2 = Float32List(p.dim),
    hb = Float32List(p.hiddenDim),
    hb2 = Float32List(p.hiddenDim),
    // ИСПРАВЛЕНИЕ: `xq` должен быть размером allHeadsDim, так как это максимальный размер, который в него квантуется.
    xq = QuantizedTensor.empty(p.nHeads * p.headDim, (p.nHeads * p.headDim) ~/ GS),
    hq = QuantizedTensor.empty(p.hiddenDim, p.hiddenDim ~/ GS),
    q = Float32List(p.nHeads * p.headDim),
    att = Float32List(p.nHeads * p.seqLen),
    logits = Float32List(p.vocabSize),
    keyCache = Float32List(p.nLayers * p.seqLen * (p.nKVHeads * p.headDim)),
    valueCache = Float32List(p.nLayers * p.seqLen * (p.nKVHeads * p.headDim));
}

class Transformer {
  final Config config;
  final TransformerWeights weights;
  final RunState state;
  Transformer(this.config, this.weights) : state = RunState(config);
}


// ----------------------------------------------------------------------------
// Функции для чтения из файла и (де)квантования
// ----------------------------------------------------------------------------

int _offset = 0;
Config _readConfig(ByteData data) {
  final config = Config(magicNumber: data.getUint32(_offset, Endian.little), version: data.getInt32(_offset + 4, Endian.little), dim: data.getInt32(_offset + 8, Endian.little), hiddenDim: data.getInt32(_offset + 12, Endian.little), nLayers: data.getInt32(_offset + 16, Endian.little), nHeads: data.getInt32(_offset + 20, Endian.little), nKVHeads: data.getInt32(_offset + 24, Endian.little), vocabSize: data.getInt32(_offset + 28, Endian.little), seqLen: data.getInt32(_offset + 32, Endian.little), headDim: data.getInt32(_offset + 36, Endian.little), sharedClassifier: data.getInt32(_offset + 40, Endian.little) == 1, groupSize: data.getInt32(_offset + 44, Endian.little));
  _offset = 256;
  return config;
}

Float32List _readFloats(ByteData data, int n) {
  final list = data.buffer.asFloat32List(_offset, n);
  _offset += n * 4;
  return list;
}

Int8List _readInt8s(ByteData data, int n) {
  final list = data.buffer.asInt8List(_offset, n);
  _offset += n;
  return list;
}

List<QuantizedTensor> _initQuantizedTensors(ByteData data, int n, int sizeEach) {
  final res = <QuantizedTensor>[];
  for (int i = 0; i < n; i++) {
    final q = _readInt8s(data, sizeEach);
    final s = _readFloats(data, sizeEach ~/ GS);
    res.add(QuantizedTensor(q: q, s: s));
  }
  return res;
}

void dequantize(QuantizedTensor qx, Float32List x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = qx.q[i] * qx.s[i ~/ GS];
  }
}

void quantize(QuantizedTensor qx, Float32List x, int n) {
  final numGroups = n ~/ GS;
  const qMax = 127.0;
  for (int group = 0; group < numGroups; group++) {
    double wmax = 0.0;
    final base = group * GS;
    for (int i = 0; i < GS; i++) {
      final val = x[base + i].abs();
      if (val > wmax) {
        wmax = val;
      }
    }
    final scale = wmax / qMax;
    qx.s[group] = (scale == 0) ? 1.0 : scale.toDouble();
    for (int i = 0; i < GS; i++) {
      final quantValue = x[base + i] / qx.s[group];
      qx.q[base + i] = quantValue.round();
    }
  }
}

Future<Transformer> newTransformer(String checkpointPath, int ctxLength) async {
  _offset = 0;
  final fileData = await File(checkpointPath).readAsBytes();
  final data = fileData.buffer.asByteData();
  final config = _readConfig(data);
  if (config.magicNumber != 0x616a6331) throw Exception('Not a qwen3.c checkpoint');
  if (config.version != 1) throw Exception('Wrong version');
  
  GS = config.groupSize;
  if (ctxLength > 0 && ctxLength <= config.seqLen) {
    config.seqLen = ctxLength;
  }
  
  final w = TransformerWeights();
  final dim = config.dim, nLayers = config.nLayers, vocabSize = config.vocabSize, hiddenDim = config.hiddenDim, nHeads = config.nHeads, nKVHeads = config.nKVHeads, headDim = config.headDim, allHeadsDim = nHeads * headDim, kvDim = nKVHeads * headDim;
  w.rmsAttWeight = _readFloats(data, nLayers * dim);
  w.rmsFfnWeight = _readFloats(data, nLayers * dim);
  w.rmsFinalWeight = _readFloats(data, dim);
  w.qlnWeights = _readFloats(data, nLayers * headDim);
  w.klnWeights = _readFloats(data, nLayers * headDim);
  w.qTokens = _initQuantizedTensors(data, 1, vocabSize * dim)[0];
  w.tokenEmbeddingTable = Float32List(vocabSize * dim);
  dequantize(w.qTokens, w.tokenEmbeddingTable, vocabSize * dim);
  w.wq = _initQuantizedTensors(data, nLayers, dim * allHeadsDim);
  w.wk = _initQuantizedTensors(data, nLayers, dim * kvDim);
  w.wv = _initQuantizedTensors(data, nLayers, dim * kvDim);
  w.wo = _initQuantizedTensors(data, nLayers, allHeadsDim * dim);
  w.w1 = _initQuantizedTensors(data, nLayers, dim * hiddenDim);
  w.w2 = _initQuantizedTensors(data, nLayers, hiddenDim * dim);
  w.w3 = _initQuantizedTensors(data, nLayers, dim * hiddenDim);
  if (config.sharedClassifier) {
    w.wCls = w.qTokens;
  } else {
    w.wCls = _initQuantizedTensors(data, 1, dim * vocabSize)[0];
  }
  return Transformer(config, w);
}

// ----------------------------------------------------------------------------
// Математические функции
// ----------------------------------------------------------------------------
void rmsnorm(Float32List o, Float32List x, Float32List weight, {int oOffset = 0, int xOffset = 0, int wOffset = 0, required int size}) {
  double ss = 0.0;
  for (int j = 0; j < size; j++) {
    final val = x[xOffset + j];
    ss += val * val;
  }
  ss /= size;
  ss += 1e-6;
  ss = 1.0 / math.sqrt(ss);
  for (int j = 0; j < size; j++) {
    o[oOffset + j] = weight[wOffset + j] * (ss * x[xOffset + j]);
  }
}

void softmax(Float32List x, {int offset = 0, required int size}) {
  if (size == 0) return;
  double maxVal = x[offset];
  for (int i = 1; i < size; i++) {
    if (x[offset + i] > maxVal) {
      maxVal = x[offset + i];
    }
  }
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    x[offset + i] = math.exp(x[offset + i] - maxVal);
    sum += x[offset + i];
  }
  for (int i = 0; i < size; i++) {
    x[offset + i] /= sum;
  }
}

void matmul(Float32List xout, QuantizedTensor x, QuantizedTensor w, int n, int d, {int xoutOffset = 0}) {
  for (int i = 0; i < d; i++) {
    double val = 0;
    final inBase = i * n;
    for (int j = 0; j <= n - GS; j += GS) {
      int ival = 0;
      for (int k = 0; k < GS; k++) {
        ival += x.q[j + k] * w.q[inBase + j + k];
      }
      val += ival * w.s[(inBase + j) ~/ GS] * x.s[j ~/ GS];
    }
    xout[xoutOffset + i] = val;
  }
}

// ----------------------------------------------------------------------------
// Прямой проход (однопоточный)
// ----------------------------------------------------------------------------
Future<Float32List> forward(Transformer t, int token, int pos) async {
  return Future.value(() {
    final p = t.config;
    final w = t.weights;
    final s = t.state;
    final x = s.x;
    final dim = p.dim;
    final kvDim = p.nKVHeads * p.headDim;
    final kvMul = p.nHeads ~/ p.nKVHeads;
    final hiddenDim = p.hiddenDim;
    final allHeadsDim = p.nHeads * p.headDim;
    final headDim = p.headDim;
    
    x.setRange(0, dim, w.tokenEmbeddingTable.sublist(token * dim, (token + 1) * dim));
    
    for (int l = 0; l < p.nLayers; l++) {
      rmsnorm(s.xb, x, w.rmsAttWeight, wOffset: l * dim, size: dim);
      quantize(s.xq, s.xb, dim);
      
      final kCacheLOff = l * p.seqLen * kvDim;
      final vCacheLOff = l * p.seqLen * kvDim;
      
      matmul(s.q, s.xq, w.wq[l], dim, allHeadsDim);
      matmul(s.keyCache, s.xq, w.wk[l], dim, kvDim, xoutOffset: kCacheLOff + pos * kvDim);
      matmul(s.valueCache, s.xq, w.wv[l], dim, kvDim, xoutOffset: vCacheLOff + pos * kvDim);
      
      for (int h = 0; h < p.nHeads; h++) {
        final qOffset = h * headDim;
        rmsnorm(s.q, s.q, w.qlnWeights, oOffset: qOffset, xOffset: qOffset, wOffset: l * headDim, size: headDim);
        for (int j = 0; j < headDim ~/ 2; j++) {
          final freq = math.pow(1e6, -j / (headDim / 2));
          final cosFreq = math.cos(pos * freq);
          final sinFreq = math.sin(pos * freq);
          final realPart = s.q[qOffset + j];
          final imagPart = s.q[qOffset + j + headDim ~/ 2];
          s.q[qOffset + j] = realPart * cosFreq - imagPart * sinFreq;
          s.q[qOffset + j + headDim ~/ 2] = realPart * sinFreq + imagPart * cosFreq;
        }
      }
      
      for (int h = 0; h < p.nKVHeads; h++) {
        final kOffset = kCacheLOff + pos * kvDim + h * headDim;
        rmsnorm(s.keyCache, s.keyCache, w.klnWeights, oOffset: kOffset, xOffset: kOffset, wOffset: l * headDim, size: headDim);
        for (int j = 0; j < headDim ~/ 2; j++) {
          final freq = math.pow(1e6, -j / (headDim / 2));
          final cosFreq = math.cos(pos * freq);
          final sinFreq = math.sin(pos * freq);
          final realPart = s.keyCache[kOffset + j];
          final imagPart = s.keyCache[kOffset + j + headDim ~/ 2];
          s.keyCache[kOffset + j] = realPart * cosFreq - imagPart * sinFreq;
          s.keyCache[kOffset + j + headDim ~/ 2] = realPart * sinFreq + imagPart * cosFreq;
        }
      }
      
      for (int h = 0; h < p.nHeads; h++) {
        final q_h_offset = h * headDim;
        final att_h_offset = h * p.seqLen;
        for (int t = 0; t <= pos; t++) {
          final k_t_offset = kCacheLOff + t * kvDim + (h ~/ kvMul) * headDim;
          double score = 0.0;
          for (int i = 0; i < headDim; i++) {
            score += s.q[q_h_offset + i] * s.keyCache[k_t_offset + i];
          }
          s.att[att_h_offset + t] = score / math.sqrt(headDim);
        }
        softmax(s.att, offset: att_h_offset, size: pos + 1);
        final xb_h_offset = h * headDim;
        for (int i = 0; i < headDim; i++) s.xb[xb_h_offset + i] = 0;
        for (int t = 0; t <= pos; t++) {
          final v_t_offset = vCacheLOff + t * kvDim + (h ~/ kvMul) * headDim;
          final a = s.att[att_h_offset + t];
          for (int i = 0; i < headDim; i++) {
            s.xb[xb_h_offset + i] += a * s.valueCache[v_t_offset + i];
          }
        }
      }
      
      quantize(s.xq, s.xb, allHeadsDim);
      matmul(s.xb2, s.xq, w.wo[l], allHeadsDim, dim);
      
      for (int i = 0; i < dim; i++) { x[i] += s.xb2[i]; }
      
      rmsnorm(s.xb, x, w.rmsFfnWeight, wOffset: l * dim, size: dim);
      quantize(s.xq, s.xb, dim);
      matmul(s.hb, s.xq, w.w1[l], dim, hiddenDim);
      matmul(s.hb2, s.xq, w.w3[l], dim, hiddenDim);
      
      for (int i = 0; i < hiddenDim; i++) {
        double val = s.hb[i];
        val *= (1.0 / (1.0 + math.exp(-val)));
        val *= s.hb2[i];
        s.hb[i] = val;
      }
      
      quantize(s.hq, s.hb, hiddenDim);
      matmul(s.xb, s.hq, w.w2[l], hiddenDim, dim);
      
      for (int i = 0; i < dim; i++) { x[i] += s.xb[i]; }
    }
    
    rmsnorm(x, x, w.rmsFinalWeight, size: dim);
    quantize(s.xq, x, dim);
    matmul(s.logits, s.xq, w.wCls, dim, p.vocabSize);
    
    return s.logits;
  }());
}

// ----------------------------------------------------------------------------
// Токенизатор
// ----------------------------------------------------------------------------
class Tokenizer {
  final List<String> vocab;
  final Float32List mergeScores;
  final int vocabSize, maxTokenLength, bosTokenID, eosTokenID;
  String promptTemplate, systemPromptTemplate;
  final Map<String, int> vocabMap;

  Tokenizer({
    required this.vocab,
    required this.mergeScores,
    required this.vocabSize,
    required this.maxTokenLength,
    required this.bosTokenID,
    required this.eosTokenID,
    required this.promptTemplate,
    required this.systemPromptTemplate
  }) : vocabMap = {for (var i = 0; i < vocab.length; i++) vocab[i]: i};

  static Future<String> _loadTemplate(String path, String suffix) async {
    try {
      final data = await File(path + suffix).readAsBytes();
      int len = data.length;
      while (len > 0 && data[len - 1] == 0) len--;
      return String.fromCharCodes(data.sublist(0, len));
    } catch (e) {
      throw Exception("couldn't load $path$suffix: $e");
    }
  }

  static Future<Tokenizer> newTokenizer(String path, int vocabSize, bool enableThinking) async {
    final tokenizerPath = '$path.tokenizer';
    final file = File(tokenizerPath);
    if (!await file.exists()) throw Exception("can't load $tokenizerPath");
    
    final data = (await file.readAsBytes()).buffer.asByteData();
    int offset = 0;
    final maxTokenLength = data.getUint32(offset, Endian.little); offset += 4;
    final bosTokenID = data.getUint32(offset, Endian.little); offset += 4;
    final eosTokenID = data.getUint32(offset, Endian.little); offset += 4;
    
    final vocab = List<String>.filled(vocabSize, "");
    final mergeScores = Float32List(vocabSize);
    for (int i = 0; i < vocabSize; i++) {
      if (offset + 4 > data.lengthInBytes) break;
      mergeScores[i] = data.getFloat32(offset, Endian.little); offset += 4;
      if (offset + 4 > data.lengthInBytes) break;
      final len = data.getInt32(offset, Endian.little); offset += 4;
      if (offset + len > data.lengthInBytes) break;
      vocab[i] = String.fromCharCodes(data.buffer.asUint8List(offset, len));
      offset += len;
    }
    
    var promptSuffix = ".template", systemPromptSuffix = ".template.with-system";
    if (enableThinking) {
      promptSuffix += ".with-thinking";
      systemPromptSuffix += ".with-thinking";
    }
    
    return Tokenizer(
      vocab: vocab,
      mergeScores: mergeScores,
      vocabSize: vocabSize,
      maxTokenLength: maxTokenLength,
      bosTokenID: bosTokenID,
      eosTokenID: eosTokenID,
      promptTemplate: await _loadTemplate(path, promptSuffix),
      systemPromptTemplate: await _loadTemplate(path, systemPromptSuffix)
    );
  }

  int strLookup(String s) => vocabMap[s] ?? -1;
  
  List<int> encode(String text) {
    var tokens = <int>[];
    for (int i = 0; i < text.length;) {
      if (text[i] == '<') {
        final end = text.indexOf('>', i);
        if (end != -1) {
          final specialToken = text.substring(i, end + 1);
          final id = strLookup(specialToken);
          if (id != -1) {
            tokens.add(id);
            i += specialToken.length;
            continue;
          }
        }
      }
      final id = strLookup(text[i]);
      if (id != -1) {
        tokens.add(id);
      } else {
        stderr.writeln("Warning: unknown char ${text[i]}");
      }
      i++;
    }
    while (true) {
      double bestScore = -1e10;
      int bestId = -1, bestIdx = -1;
      for (int i = 0; i < tokens.length - 1; i++) {
        final merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
        final id = strLookup(merged);
        if (id != -1 && mergeScores[id] > bestScore) {
          bestScore = mergeScores[id];
          bestId = id;
          bestIdx = i;
        }
      }
      if (bestIdx == -1) break;
      tokens[bestIdx] = bestId;
      tokens.removeAt(bestIdx + 1);
    }
    return tokens;
  }

  String decode(int token) => vocab[token];
}

// ----------------------------------------------------------------------------
// Семплер
// ----------------------------------------------------------------------------
class ProbIndex {
  double prob;
  int index;
  ProbIndex(this.prob, this.index);
}

class Sampler {
  final int vocabSize;
  final List<ProbIndex> probIndex;
  final double temperature, topp;
  final math.Random rng;
  
  Sampler({required this.vocabSize, required this.temperature, required this.topp, required int seed}) : 
    probIndex = List.generate(vocabSize, (i) => ProbIndex(0.0, 0)),
    rng = math.Random(seed);
  
  int sample(Float32List logits) {
    if (temperature == 0.0) return _argmax(logits);
    for (int i = 0; i < logits.length; i++) logits[i] /= temperature;
    softmax(logits, size: logits.length);
    final coin = rng.nextDouble();
    return (topp <= 0 || topp >= 1) ? _mult(logits, coin) : _topp(logits, topp, coin);
  }

  int _argmax(Float32List p) {
    int maxI = 0;
    double maxP = p[0];
    for (int i = 1; i < p.length; i++) {
      if (p[i] > maxP) {
        maxI = i;
        maxP = p[i];
      }
    }
    return maxI;
  }

  int _mult(Float32List p, double c) {
    double cdf = 0.0;
    for (int i = 0; i < p.length; i++) {
      cdf += p[i];
      if (c < cdf) return i;
    }
    return p.length - 1;
  }

  int _topp(Float32List p, double topp, double c) {
    final n = p.length;
    final cutoff = (1.0 - topp) / (n - 1);
    int n0 = 0;
    for (int i = 0; i < n; i++) {
      if (p[i] >= cutoff) {
        probIndex[n0].index = i;
        probIndex[n0].prob = p[i];
        n0++;
      }
    }
    probIndex.sublist(0, n0).sort((a, b) => b.prob.compareTo(a.prob));
    double cprob = 0.0;
    int last = n0 - 1;
    for (int i = 0; i < n0; i++) {
      cprob += probIndex[i].prob;
      if (cprob > topp) {
        last = i;
        break;
      }
    }
    final r = c * cprob;
    double cdf = 0.0;
    for (int i = 0; i <= last; i++) {
      cdf += probIndex[i].prob;
      if (r < cdf) return probIndex[i].index;
    }
    return probIndex[last].index;
  }
}

// ----------------------------------------------------------------------------
// Цикл генерации
// ----------------------------------------------------------------------------
Future<void> generate(Transformer t, Tokenizer tokenizer, Sampler sampler, String prompt) async {
  if (prompt.isEmpty) {
    stderr.writeln("Please provide a prompt using -i");
    exit(1);
  }
  final pt = tokenizer.encode(prompt);
  if (pt.isEmpty) return;
  
  int token = pt[0], pos = 0;
  final sw = Stopwatch()..start();
  
  while (pos < t.config.seqLen) {
    final logits = await forward(t, token, pos);
    int next = (pos < pt.length - 1) ? pt[pos + 1] : sampler.sample(logits);
    stdout.write(tokenizer.decode(token));
    
    if (++pos >= t.config.seqLen) break;
    token = next;
    if (pos >= pt.length && (next == tokenizer.eosTokenID)) break;
  }
  sw.stop();
  stdout.writeln();
  if (sw.elapsedMilliseconds > 0) {
    stdout.writeln('Achieved: ${(pos / sw.elapsedMilliseconds * 1000).toStringAsFixed(2)} tokens/sec');
  }
}

Future<void> chat(Transformer t, Tokenizer tokenizer, Sampler sampler, String? cliUserPrompt, String? systemPrompt) async {
  int pos = 0;
  bool userTurn = true;
  final stopwatch = Stopwatch();
  
  while (true) {
    if (pos >= t.config.seqLen) {
      stdout.writeln("\n(context full, clearing)");
      pos = 0;
      userTurn = true;
    }
    
    List<int> pt = [];
    if (userTurn) {
      String up;
      if (cliUserPrompt != null && pos == 0) {
        up = cliUserPrompt;
      } else {
        stdout.write("\n> ");
        up = stdin.readLineSync() ?? "";
        if (up.trim().isEmpty) break;
      }
      String rp = (pos == 0 && systemPrompt != null && systemPrompt.isNotEmpty)
        ? tokenizer.systemPromptTemplate.replaceAll('%s', systemPrompt).replaceFirst('%s', up)
        : tokenizer.promptTemplate.replaceFirst('%s', up);
      pt = tokenizer.encode(rp);
      userTurn = false;
      stdout.write("\n< ");
    }
    
    if (pt.isEmpty) continue;
    
    // Обработка промпта
    int token = pt[0];
    if (pt.length > 1) {
       token = pt.removeAt(0);
       while (pt.isNotEmpty) {
         await forward(t, token, pos);
         if (++pos >= t.config.seqLen) break;
         token = pt.removeAt(0);
       }
    }
    if (pos >= t.config.seqLen) continue;
    
    // Генерация ответа
    stopwatch.reset();
    stopwatch.start();
    int tokensGenerated = 0;
    while (pos < t.config.seqLen) {
      final logits = await forward(t, token, pos);
      final next = sampler.sample(logits);
      
      if (next == tokenizer.eosTokenID) {
        userTurn = true;
        break;
      }
      
      stdout.write(tokenizer.decode(next));
      tokensGenerated++;
      
      if (++pos >= t.config.seqLen) break;
      token = next;
    }
    
    stopwatch.stop();
    if (userTurn && stopwatch.elapsedMilliseconds > 0) {
      final tps = (tokensGenerated / stopwatch.elapsedMilliseconds * 1000).toStringAsFixed(2);
      stdout.writeln('\n(generated $tokensGenerated tokens at $tps tokens/sec)');
    }
  }
}

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------
void main(List<String> args) async {
  final parser = ArgParser()
    ..addOption('t', defaultsTo: '1.0', help: "temperature")
    ..addOption('p', defaultsTo: '0.9', help: "top-p sampling")
    ..addOption('s', defaultsTo: '0', help: "random seed")
    ..addOption('c', defaultsTo: '0', help: "context window size")
    ..addOption('m', defaultsTo: 'chat', help: "mode: generate|chat")
    ..addOption('i', help: "input prompt")
    ..addOption('y', help: "system prompt")
    ..addOption('r', defaultsTo: '0', help: "reasoning mode (0 or 1)");
    
  final results = parser.parse(args);
  if (results.rest.isEmpty) {
    print("Usage: runq <checkpoint> [options]\n${parser.usage}");
    exit(1);
  }
  
  final checkpointPath = results.rest.first;
  final temp = double.parse(results['t']);
  final topp = double.parse(results['p']);
  int seed = int.parse(results['s']);
  if (seed <= 0) seed = DateTime.now().millisecondsSinceEpoch;
  final ctxLength = int.parse(results['c']);
  final mode = results['m'];
  final prompt = results['i'];
  final systemPrompt = results['y'];
  final enableThinking = results['r'] == '1';

  try {
    final transformer = await newTransformer(checkpointPath, ctxLength);
    final tokenizer = await Tokenizer.newTokenizer(checkpointPath, transformer.config.vocabSize, enableThinking);
    final sampler = Sampler(vocabSize: transformer.config.vocabSize, temperature: temp, topp: topp, seed: seed);

    if (prompt == null && mode == 'chat') {
      print(transformer.config);
    }

    if (mode == 'generate') {
      await generate(transformer, tokenizer, sampler, prompt ?? "");
    } else {
      await chat(transformer, tokenizer, sampler, prompt, systemPrompt);
    }
  } catch (e, st) {
      stderr.writeln("An error occurred: $e");
      stderr.writeln("Stack trace:\n$st");
      exit(1);
  }
}