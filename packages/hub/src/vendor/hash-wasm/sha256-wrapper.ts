import WasmModule from "./sha256";

const BUFFER_MAX_SIZE = 8 * 1024 * 1024;

export async function createSHA256(isInsideWorker = false): Promise<{
	init(): void;
	update(data: Uint8Array): void;
	digest(method: "hex"): string;
}> {
	const wasm: Awaited<ReturnType<typeof WasmModule>> = isInsideWorker
		? // @ts-expect-error WasmModule will be populated inside self object
		  await self["WasmModule"]()
		: await WasmModule();
	const heap = wasm.HEAPU8.subarray(wasm._GetBufferPtr());
	return {
		init() {
			wasm._Hash_Init(256);
		},
		update(data: Uint8Array) {
			let byteUsed = 0;
			while (byteUsed < data.byteLength) {
				const bytesLeft = data.byteLength - byteUsed;
				const length = bytesLeft < BUFFER_MAX_SIZE ? bytesLeft : BUFFER_MAX_SIZE;
				heap.set(data.subarray(byteUsed, length));
				wasm._Hash_Update(length);
				byteUsed += length;
			}
		},
		digest(method: "hex") {
			if (method !== "hex") {
				throw new Error("Only digest hex is supported");
			}
			wasm._Hash_Final();
			const result = Array.from(heap.slice(0, 32));
			return result.map((b) => b.toString(16).padStart(2, "0")).join("");
		},
	};
}

export function createSHA256WorkerCode(): string {
	return `
    const _scriptDir = "";
    self.WasmModule = ${WasmModule.toString()};

    ${createSHA256.toString()};

    self.addEventListener('message', async (event) => {
      const { file } = event.data;
      const sha256 = await createSHA256(true);
      sha256.init();
      const reader = file.stream().getReader();
      const total = file.size;
      let bytesDone = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        sha256.update(value);
        bytesDone += value.length;
        postMessage({ progress: bytesDone / total });
      }
      postMessage({ sha256: sha256.digest('hex') });
    });
  `;
}
