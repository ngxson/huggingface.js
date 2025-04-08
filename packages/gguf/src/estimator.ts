import { GGMLQuantizationType } from "@huggingface/tasks";
import { GGUFParseOutput } from "./types";
import { GGML_QUANT_SIZES } from "./quant-descriptions";

const TO_MEGA = 1e-6;
const TO_GIGA = 1e-9;

export interface RuntimeRequirementEstimation {
  memory: {
    perToken: number; // in megabytes
    weight: number; // in megabytes
  },
  compute: {
    gFloatOps: number; // in gigaflops, per token
  }
}

export interface RuntimeConfig {
  kvTypeK: GGMLQuantizationType,
  kvTypeV: GGMLQuantizationType,
}

export function estimateRuntimeRequirement(
  config: RuntimeConfig,
  metadata: GGUFParseOutput<{ strict: false }>["metadata"],
  tensorInfos?: GGUFParseOutput<{ strict: false }>["tensorInfos"],
): RuntimeRequirementEstimation {
  const kvTypeK = config.kvTypeK ?? GGMLQuantizationType.F16;
  const kvTypeV = config.kvTypeV ?? GGMLQuantizationType.F16;

  // for calculating memory usage
  const arch = metadata["general.architecture"] ?? "unknown";
	const n_embd = (metadata[`${arch}.embedding_length`] as number) ?? 0;
	const n_head = (metadata[`${arch}.attention.head_count`] as number) ?? 0;
	const n_embd_head_k = (metadata[`${arch}.attention.key_length`] as number) ?? n_embd / n_head;
	const n_embd_head_v = (metadata[`${arch}.attention.value_length`] as number) ?? n_embd / n_head;
	const n_head_kv = (metadata[`${arch}.attention.head_count_kv`] as number[] | number) ?? [];
	const n_layer = (metadata[`${arch}.block_count`] as number) ?? 0;

	if (arch.startsWith("mamba") || arch.startsWith("rwkv")) {
		throw new Error(`Memory usage estimation for arch "${arch}" is not supported`);
	}

  const n_head_kv_arr = Array(n_layer).fill(n_head);
	if (Array.isArray(n_head_kv)) {
		for (let i = 0; i < n_layer; i++) {
			if (n_head_kv[i]) {
				n_head_kv_arr[i] = n_head_kv[i];
			}
		}
	} else {
		for (let i = 0; i < n_layer; i++) {
			n_head_kv_arr[i] = n_head_kv;
		}
	}

  // calculate total bytes for K and V
	let cacheMBytesK = 0;
	let cacheMBytesV = 0;
	for (let i = 0; i < n_layer; i++) {
		const n_embd_k_gqa = n_embd_head_k * n_head_kv_arr[i];
		const n_embd_v_gqa = n_embd_head_v * n_head_kv_arr[i];
		cacheMBytesK += n_embd_k_gqa * (GGML_QUANT_SIZES[kvTypeK] / 8) * TO_MEGA;
		cacheMBytesV += n_embd_v_gqa * (GGML_QUANT_SIZES[kvTypeV] / 8) * TO_MEGA;
	}

  // calculate total bytes for model weight
  let mBytesWeight = 0;
  for (const tensorInfo of tensorInfos || []) {
    const nElem = Number(tensorInfo.shape.reduce((a, b) => a * b, 1n));
    const tensorSizeInBytes = nElem * (GGML_QUANT_SIZES[tensorInfo.dtype] / 8);
    mBytesWeight += tensorSizeInBytes * TO_MEGA;
  }

  // for calculating compute requirement
  const n_expert = (metadata[`${arch}.expert_count`] as number) ?? 1;
  const n_expert_used = (metadata[`${arch}.expert_used_count`] as number) ?? 1;
  const n_ff = (metadata[`${arch}.feed_forward_length`] as number) ?? 1;
  const n_vocab = (metadata[`${arch}.vocab_size`] as number) ?? 1;

  // for MoE models, we need to calculate the ratio of used/total in FFN layer (must be >= 1)
  // for non-MoE models, this ratio is 1
  const moe_ratio = Math.max(1, Math.min(1, n_expert_used) / Math.min(1, n_expert));

  // number of float ops required to multiply two matrices of shape (x, y) and (y, z)
  const floatOpsMatMul = (x: number, y: number, z: number) => 2 * x * y * z;

  // calculate total float ops per token
  let gFloatOps = 0;
  for (let i = 0; i < n_layer; i++) {
    let floatOpsLayer = 0;
    // various element-wise operations like layer norm, bias, etc.
    floatOpsLayer += 1000 * n_embd;
    floatOpsLayer += 4 * 3 * n_embd * n_embd; // self attention (Q, K, V projection)
    floatOpsLayer += 4 * 2 * n_embd * n_ff * moe_ratio; // feed forward
    gFloatOps += floatOpsLayer * TO_GIGA;
  }
  gFloatOps += floatOpsMatMul(n_embd, n_vocab, 1) * TO_GIGA; // output logits

  return {
    memory: {
      perToken: cacheMBytesK + cacheMBytesV,
      weight: mBytesWeight,
    },
    compute: {
      gFloatOps,
    },
  };
}
